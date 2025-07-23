import os
import pickle
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
import chemprop
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from chemprop import featurizers
from loguru import logger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchmetrics.classification import Accuracy
from torcheval.metrics.functional import multilabel_accuracy

from quarc.cli.train_args import TrainArgs
from quarc.data.gnn_datasets import (
    AugmentedAgentReactionDatasetWithReactionClass,
    AgentReactionDatasetWithReactionClass,
    GNNBinnedTemperatureDataset,
    GNNBinnedReactantAmountDataset,
    GNNBinnedAgentAmountOneShotDataset,
)
from quarc.models.gnn_models import (
    TemperatureGNN,
    ReactantAmountGNN,
    AgentAmountOneshotGNN,
    AgentGNNWithReactionClass,
)
from quarc.models.modules.gnn_heads import (
    GNNAgentHead,
    GNNTemperatureHead,
    GNNReactantAmountHead,
    GNNAgentAmountHead,
    GNNAgentHeadWithReactionClass,
)
from quarc.models.callbacks import GNNGreedySearchCallback
from quarc.data.gnn_dataloader import build_dataloader_agent
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
def train_stage_1_model_with_reaction_class(args: TrainArgs):
    pl.seed_everything(args.seed, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Ensure distributed initialization happens before anything else
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=local_rank,
        )
    torch.cuda.set_device(local_rank)

    save_dir = cfg.models_dir / "GNN" / f"stage{args.stage}_rxnclass"
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
        logger.info(f"data length: train: {len(train_data)}, val: {len(val_data)}")

    # Load Data
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    train_dataset = AugmentedAgentReactionDatasetWithReactionClass(
        original_data=train_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )
    val_dataset = AgentReactionDatasetWithReactionClass(
        data=val_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )

    if local_rank == 0:
        logger.info(f"train_dataset length: {len(train_dataset)}")
        logger.info(f"val_dataset length: {len(val_dataset)}")

    train_loader = build_dataloader_agent(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        distributed=True,
        persistent_workers=False,
        pin_memory=True,
    )
    val_loader = build_dataloader_agent(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        distributed=True,
        persistent_workers=False,
        pin_memory=True,
    )

    # Define Model
    fdims = featurizer.shape
    mp = chemprop.nn.BondMessagePassing(*fdims, d_h=args.graph_hidden_size, depth=args.depth)
    agg = chemprop.nn.MeanAggregation()
    predictor = GNNAgentHeadWithReactionClass(
        graph_input_dim=args.graph_hidden_size,
        agent_input_dim=len(a_enc),
        output_dim=len(a_enc),
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )
    metrics = {
        "multilabel_accuracy_exactmatch": multilabel_accuracy,
        "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
            preds, targets, criteria="hamming"
        ),
    }

    model = AgentGNNWithReactionClass(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=True,
        metrics=metrics,
        init_lr=args.init_lr,
    )

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
    greedy_search_callback = GNNGreedySearchCallback(track_batch_indices=range(len(val_loader)))
    earlystop_callback = EarlyStopping(
        monitor="val_greedy_exactmatch_accuracy", mode="max", patience=5
    )

    callbacks = [
        weights_checkpoint,
        greedy_search_callback,
        full_checkpoint,
        earlystop_callback,
    ]

    # Define the trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="cpu" if args.no_cuda else "gpu",
        devices="auto",
        strategy="ddp",
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        deterministic=True,
        sync_batchnorm=True,
        use_distributed_sampler=True,
    )

    # train the model
    checkpoint_path = args.checkpoint_path
    if checkpoint_path and local_rank == 0:
        logger.info(f"Training resume from checkpoint path: {checkpoint_path}")

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

    save_dir = cfg.models_dir / "GNN" / f"stage{args.stage}"
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

    train_data_path = cfg.processed_data_dir / "stage2" / "stage2_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage2" / "stage2_val.pickle"

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Loading data from {train_data_path} and {val_data_path}")
        logger.info(
            f"Stage 2 data length (filtered): train: {len(train_data)}, val: {len(val_data)}"
        )

    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    train_dataset = GNNBinnedTemperatureDataset(
        data=train_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )
    val_dataset = GNNBinnedTemperatureDataset(
        data=val_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )

    train_loader = build_dataloader_agent(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = build_dataloader_agent(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )
    # Define Model
    fdims = featurizer.shape
    mp = chemprop.nn.BondMessagePassing(*fdims, d_h=args.graph_hidden_size, depth=args.depth)
    predictor = GNNTemperatureHead(
        graph_input_dim=args.graph_hidden_size,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,  # default bins (-100,201,10)-> 30 ranges + 2 boundaries. Nothing expected in lower bound (0), but 200 will fall in upper bound bins (31)
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )
    agg = chemprop.nn.MeanAggregation()
    batch_norm = True
    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = TemperatureGNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=batch_norm,
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
        sync_batchnorm=True,
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

    save_dir = cfg.models_dir / "GNN" / f"stage{args.stage}"
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
        logger.info(f"Loading data from {train_data_path} and {val_data_path}")
        logger.info(f"Stage 3 data length: train: {len(train_data)}, val: {len(val_data)}")

    # Setup datasets and dataloaders
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    fp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=args.FP_radius, fpSize=args.FP_length
    )

    train_dataset = GNNBinnedReactantAmountDataset(
        data=train_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
        morgan_generator=fp_gen,
    )
    val_dataset = GNNBinnedReactantAmountDataset(
        data=val_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
        morgan_generator=fp_gen,
    )

    train_loader = build_dataloader_agent(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = build_dataloader_agent(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )

    # Define Model
    fdims = featurizer.shape
    mp = chemprop.nn.BondMessagePassing(*fdims, d_h=args.graph_hidden_size, depth=args.depth)
    predictor = GNNReactantAmountHead(
        graph_input_dim=args.graph_hidden_size,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )
    agg = chemprop.nn.MeanAggregation()
    batch_norm = True

    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = ReactantAmountGNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=batch_norm,
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
        sync_batchnorm=True,
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
    save_dir = cfg.models_dir / "GNN" / f"stage{args.stage}"
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
    train_data_path = cfg.processed_data_dir / "stage4" / "stage4_train.pickle"
    val_data_path = cfg.processed_data_dir / "stage4" / "stage4_val.pickle"

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_data_path, "rb") as f:
        val_data = pickle.load(f)

    if local_rank == 0:
        logger.info(f"Loading data from {train_data_path} and {val_data_path}")
        logger.info(f"Stage 4 data length: train: {len(train_data)}, val: {len(val_data)}")

    # Setup datasets and dataloaders
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

    train_dataset = GNNBinnedAgentAmountOneShotDataset(
        data=train_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )
    val_dataset = GNNBinnedAgentAmountOneShotDataset(
        data=val_data,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        featurizer=featurizer,
    )

    train_loader = build_dataloader_agent(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = build_dataloader_agent(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        classification=True,
        distributed=True if world_size > 1 else False,
        persistent_workers=True,
        pin_memory=True,
    )

    # Define Model
    fdims = featurizer.shape
    mp = chemprop.nn.BondMessagePassing(*fdims, d_h=args.graph_hidden_size, depth=args.depth)
    predictor = GNNAgentAmountHead(
        graph_dim=args.graph_hidden_size,
        agent_input_dim=len(a_enc),
        output_dim=args.output_size,
        hidden_dim=args.hidden_size,
        n_blocks=args.n_blocks,
    )
    agg = chemprop.nn.MeanAggregation()
    batch_norm = True

    metrics = {
        "accuracy": Accuracy(
            task="multiclass", average="micro", num_classes=args.output_size, ignore_index=0
        ),
        "accuracy_macro": Accuracy(
            task="multiclass", average="macro", num_classes=args.output_size, ignore_index=0
        ),
    }

    model = AgentAmountOneshotGNN(
        message_passing=mp,
        agg=agg,
        predictor=predictor,
        batch_norm=batch_norm,
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
        sync_batchnorm=True,
        use_distributed_sampler=True if world_size > 1 else False,
    )

    checkpoint_path = args.checkpoint_path

    # Train model
    if checkpoint_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)


def train_gnn(arguments=None):
    """Main training function that delegates to appropriate stage."""

    args = TrainArgs().parse_args(arguments) if arguments else TrainArgs().parse_args()
    stage = args.stage

    if stage == 1:
        train_stage_1_model_with_reaction_class(args)
    elif stage == 2:
        train_stage_2_model(args)
    elif stage == 3:
        train_stage_3_model(args)
    elif stage == 4:
        train_stage_4_model_oneshot(args)
    else:
        raise ValueError(f"Invalid stage {stage}")


if __name__ == "__main__":
    train_gnn()
