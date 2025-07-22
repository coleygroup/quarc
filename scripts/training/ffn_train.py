import os
import pickle
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml

from rdkit import Chem
from loguru import logger
from torch.utils.data import DistributedSampler
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchmetrics.classification import Accuracy
from torcheval.metrics.functional import multilabel_accuracy
from torch.utils.data import DataLoader
from quarc.cli.train_args import TrainArgs

from quarc.data.ffn_datasets import (
    AugmentedAgentsDataset,
    AgentsDatasetWithReactionClass,
    BinnedTemperatureDataset,
    BinnedReactantAmountDataset,
    BinnedAgentAmoutOneshot,
)
from quarc.models.ffn_models import (
    AgentFFNWithReactionClass,
    TemperatureFFN,
    ReactantAmountFFN,
    AgentAmountFFN,
)
from quarc.models.modules.ffn_heads import (
    FFNAgentHeadWithReactionClass,
    FFNTemperatureHead,
    FFNReactantAmountHead,
    FFNAgentAmountHead,
)
from quarc.models.callbacks import FFNGreedySearchCallback
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.settings import load as load_settings

cfg = load_settings()

torch.set_float32_matmul_precision("medium")


@rank_zero_only
def log_args(args, tb_path):
    yaml_args = yaml.dump(args.as_dict(), indent=2, default_flow_style=False)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    with open(f"{tb_path}/args.yaml", "w") as fp:
        fp.write(yaml_args)
    logger.info(f"Args:\n{yaml_args}")


# Stage 1: Agent (with reaction class)
def train_stage_1_model(args: TrainArgs):
    pl.seed_everything(args.seed, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )
    torch.cuda.set_device(local_rank)

    save_dir = cfg.models_dir / "FFN" / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir
        log_args(args, tb_path)
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir

    a_enc = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )
    a_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )

    train_data_path = cfg.processed_data_dir / "stage1" / "stage1_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage1" / "stage1_val.pickle"
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Stage 1data length: train: {len(train_data)}, val: {len(val_data)}")

    # Load Data
    train_dataset = AugmentedAgentsDataset(
        original_data=train_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        sample_weighting="pascal",
        fp_radius=args.FP_radius,
        fp_length=args.FP_length,
    )
    val_dataset = AgentsDatasetWithReactionClass(
        data=val_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
    )

    if world_size > 1:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            persistent_workers=False,
            pin_memory=True,
            sampler=DistributedSampler(train_dataset, shuffle=True),
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            persistent_workers=False,
            pin_memory=True,
            sampler=DistributedSampler(val_dataset, shuffle=False),
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    # Define Model
    predictor = FFNAgentHeadWithReactionClass(
        fp_dim=args.FP_length,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )
    metrics = {
        "multilabel_accuracy_exactmatch": multilabel_accuracy,
        "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
            preds, targets, criteria="hamming"
        ),
    }

    steps_per_epoch = sum(1 for _ in train_loader)
    model = AgentFFNWithReactionClass(
        predictor=predictor,
        metrics=metrics,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )

    greedy_search_callback = FFNGreedySearchCallback(track_batch_indices=range(len(val_loader)))

    weights_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="weights-{epoch}",
        save_weights_only=True,
        save_last=False,
        every_n_epochs=1,
        save_top_k=-1,
    )

    full_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="full-{epoch}",
        save_weights_only=False,
        save_last=False,
        every_n_epochs=3,
        save_top_k=-1,
    )
    earlystop_callback = EarlyStopping(
        monitor="val_greedy_exactmatch_accuracy",
        patience=5,
        mode="max",
        check_on_train_epoch_end=False,
    )

    callbacks = [
        greedy_search_callback,
        weights_checkpoint,
        full_checkpoint,
        earlystop_callback,
    ]

    # Define the trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices=1 if args.no_cuda else world_size,
        strategy="ddp" if world_size > 1 else "auto",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        sync_batchnorm=True,
        use_distributed_sampler=True if world_size > 1 else False,
    )

    # train the model
    checkpoint_path = args.checkpoint_path

    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


# Stage 2: Temperature amount
def train_stage_2_model(args: TrainArgs):
    pl.seed_everything(args.seed, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        # Initialize process group
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )

    save_dir = cfg.models_dir / "FFN" / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir
        log_args(args, tb_path)
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir

    # * New agent encoder used
    a_enc = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )

    a_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=args.FP_radius, fpSize=args.FP_length
    )
    train_data_path = cfg.processed_data_dir / "stage2" / "stage2_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage2" / "stage2_val.pickle"

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Stage 2 data length: train: {len(train_data)}, val: {len(val_data)}")

    train_dataset = BinnedTemperatureDataset(
        data=train_data,
        morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
    )
    val_dataset = BinnedTemperatureDataset(
        data=val_data,
        morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    # Define Model
    predictor = FFNTemperatureHead(
        fp_dim=args.FP_length,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,  # default bins (-100,201,10)-> 30 ranges + 2 boundaries. Nothing expected in lower bound (0), but 200 will fall in upper bound bins (31)
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )

    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = TemperatureFFN(
        predictor=predictor,
        metrics=metrics,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )

    weights_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="weights-{epoch}",
        save_weights_only=True,
        save_last=False,
        every_n_epochs=5,
        save_top_k=-1,
    )

    full_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="full-{epoch}",
        save_weights_only=False,
        save_last=False,
        every_n_epochs=10,
        save_top_k=-1,
    )
    earlystop_callback = EarlyStopping(monitor="accuracy", patience=15, mode="max")
    callbacks = [weights_checkpoint, full_checkpoint, earlystop_callback]

    # Define the trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices="auto",
        strategy="ddp" if world_size > 1 else "auto",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        deterministic=True,
        use_distributed_sampler=True if world_size > 1 else False,
    )

    # train the model
    checkpoint_path = args.checkpoint_path

    # Train model
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


# Stage 3: Reactant amount
def train_stage_3_model(args: TrainArgs):
    pl.seed_everything(args.seed, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )

    save_dir = cfg.models_dir / "FFN" / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir
        log_args(args, tb_path)
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir

    # Setup encoders and standardizers
    a_enc = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )
    a_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )

    # Load and validate data
    train_data_path = cfg.processed_data_dir / "stage3" / "stage3_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage3" / "stage3_val.pickle"

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Stage 3 data length: train: {len(train_data)}, val: {len(val_data)}")

    # Setup datasets and dataloaders
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=args.FP_radius, fpSize=args.FP_length
    )

    train_dataset = BinnedReactantAmountDataset(
        data=train_data,
        # morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        fp_radius=args.FP_radius,
        fp_length=args.FP_length,
    )
    val_dataset = BinnedReactantAmountDataset(
        data=val_data,
        # morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        fp_radius=args.FP_radius,
        fp_length=args.FP_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    # Define Model
    predictor = FFNReactantAmountHead(
        fp_dim=args.FP_length,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )

    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = ReactantAmountFFN(
        predictor=predictor,
        metrics=metrics,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )

    # Setup callbacks
    weights_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="weights-{epoch}",
        save_weights_only=True,
        save_last=False,
        every_n_epochs=5,
        save_top_k=-1,
    )
    full_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="full-{epoch}",
        save_weights_only=False,
        save_last=False,
        every_n_epochs=10,
        save_top_k=-1,
    )
    earlystop_callback = EarlyStopping(monitor="accuracy", patience=15, mode="max")
    callbacks = [weights_checkpoint, full_checkpoint, earlystop_callback]

    # Define trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices="auto",
        strategy="ddp" if world_size > 1 else "auto",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        deterministic=True,
        use_distributed_sampler=True if world_size > 1 else False,
    )

    checkpoint_path = args.checkpoint_path

    # Train model
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


# Stage 4: Agent amount
def train_stage_4_model_oneshot(args: TrainArgs):
    pl.seed_everything(args.seed, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )
    save_dir = cfg.models_dir / "FFN" / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir
        log_args(args, tb_path)
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, name=args.logger_name)
        tb_path = tb_logger.log_dir

    # Setup encoders and standardizers
    a_enc = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )
    a_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=args.FP_radius, fpSize=args.FP_length
    )
    # Load and validate data
    train_data_path = cfg.processed_data_dir / "stage4" / "stage4_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage4" / "stage4_val.pickle"

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Stage 4 data length: train: {len(train_data)}, val: {len(val_data)}")

    # Setup datasets and dataloaders
    train_dataset = BinnedAgentAmoutOneshot(
        data=train_data,
        morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
    )
    val_dataset = BinnedAgentAmoutOneshot(
        data=val_data,
        morgan_generator=fp_gen,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    # Define Model
    predictor = FFNAgentAmountHead(
        fp_dim=args.FP_length,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )

    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = AgentAmountFFN(
        predictor=predictor,
        metrics=metrics,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )

    # Setup callbacks
    weights_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="weights-{epoch}",
        save_weights_only=True,
        save_last=False,
        every_n_epochs=5,
        save_top_k=-1,
    )
    full_checkpoint = ModelCheckpoint(
        dirpath=tb_path,
        filename="full-{epoch}",
        save_weights_only=False,
        save_last=False,
        every_n_epochs=10,
        save_top_k=-1,
    )
    earlystop_callback = EarlyStopping(monitor="accuracy", patience=15, mode="max")
    callbacks = [weights_checkpoint, full_checkpoint, earlystop_callback]

    # Define trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices="auto",
        strategy="ddp" if world_size > 1 else "auto",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        deterministic=True,
        use_distributed_sampler=True if world_size > 1 else False,
    )

    checkpoint_path = args.checkpoint_path

    # Train model
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


def train_ffn(arguments=None):
    """Main training function that delegates to appropriate stage."""
    args = TrainArgs().parse_args(arguments) if arguments else TrainArgs().parse_args()
    stage = args.stage

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )

    if stage == 1:
        train_stage_1_model(args)
    elif stage == 2:
        train_stage_2_model(args)
    elif stage == 3:
        train_stage_3_model(args)
    elif stage == 4:
        train_stage_4_model_oneshot(args)
    else:
        raise ValueError(f"Invalid stage {stage}")


if __name__ == "__main__":
    train_ffn()
