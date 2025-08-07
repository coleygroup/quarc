import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger

from quarc.cli.quarc_parser import add_model_opts, add_train_opts, add_data_opts
from quarc.training.model_factory import ModelFactory
from quarc.settings import load as load_settings

cfg = load_settings()


torch.set_float32_matmul_precision("medium")


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


@rank_zero_only
def log_rank_0(message):
    logger.info(message)


@rank_zero_only
def save_args(args, checkpoint_dir):
    args_dict = vars(args)
    args_yaml = yaml.dump(args_dict, indent=2, default_flow_style=False)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(f"{checkpoint_dir}/args.yaml", "w") as f:
        f.write(args_yaml)

    log_rank_0(f"Training arguments:\n{args_yaml}")


def setup_distributed():
    local_rank = get_local_rank()
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def setup_logger(args):
    save_dir = Path(args.save_dir) / args.model_type.upper() / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir, name=args.logger_name or f"{args.model_type}_stage{args.stage}"
    )

    save_args(args, tb_logger.log_dir)

    return tb_logger


def load_stage_data(args):
    if not args.train_data_path:
        args.train_data_path = (
            cfg.processed_data_dir / f"stage{args.stage}/stage{args.stage}_train.pickle"
        )
    if not args.val_data_path:
        args.val_data_path = (
            cfg.processed_data_dir / f"stage{args.stage}/stage{args.stage}_val.pickle"
        )

    with open(args.train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(args.val_data_path, "rb") as f:
        val_data = pickle.load(f)

    log_rank_0(
        f"Loading data from {args.train_data_path} and {args.val_data_path}"
        f"train={len(train_data)}, val={len(val_data)}"
    )

    return train_data, val_data


def setup_callbacks(args, tb_logger, extra_callbacks=None):
    stage = args.stage
    tb_path = tb_logger.log_dir

    weights_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="weights-{epoch}",
        save_weights_only=True,
        save_last=False,
        every_n_epochs=1 if stage == 1 else 5,
        save_top_k=-1,
    )

    full_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="full-{epoch}",
        save_weights_only=False,
        save_last=False,
        every_n_epochs=3 if stage == 1 else 10,
        save_top_k=-1,
    )

    callbacks = [weights_checkpoint, full_checkpoint]

    if args.early_stop:
        if stage == 1:
            earlystop_callback = EarlyStopping(
                monitor="val_greedy_exactmatch_accuracy",
                patience=args.early_stop_patience,
                mode="max",
                check_on_train_epoch_end=False,
            )
        else:
            earlystop_callback = EarlyStopping(
                monitor="accuracy",
                patience=args.early_stop_patience,
                mode="max",
            )
        callbacks.append(earlystop_callback)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return callbacks


def train_stage(args):
    log_rank_0(f"Starting {args.model_type.upper()} Stage {args.stage} training")

    # setup
    pl.seed_everything(args.seed, workers=True)
    _, world_size = setup_distributed()
    tb_logger = setup_logger(args)

    # load data and model
    train_data, val_data = load_stage_data(args)
    model_factory = ModelFactory(args)
    model, train_loader, val_loader, extra_callbacks = model_factory.create_model_and_data(
        train_data, val_data
    )

    # set up trainer
    callbacks = setup_callbacks(args, tb_logger, extra_callbacks)
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices="auto",
        strategy="ddp" if world_size > 1 else "auto",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        sync_batchnorm=True,
        use_distributed_sampler=True if world_size > 1 else False,
        deterministic=True,
    )

    # train
    try:
        if args.checkpoint_path:
            log_rank_0(f"Loading checkpoint from {args.checkpoint_path}")
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint_path)
        else:
            trainer.fit(model, train_loader, val_loader)

    except Exception as e:
        if get_local_rank() == 0:
            logger.error(f"Training failed for {args.model_type} stage {args.stage}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="training")
    add_model_opts(parser)
    add_train_opts(parser)

    args, unknown = parser.parse_known_args()

    if get_local_rank() == 0:
        log_dir = cfg.logs_dir / "train" / args.model_type / f"stage{args.stage}"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{args.logger_name}_{datetime.now().strftime("%y%m%d_%H%M")}.log"

        logger.remove()
        logger.add(sys.stderr, level="INFO", colorize=True)
        logger.add(str(log_file), level="INFO")

    train_stage(args)


if __name__ == "__main__":
    main()
