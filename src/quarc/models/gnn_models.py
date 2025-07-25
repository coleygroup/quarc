from typing import Callable

import chemprop
import lightning.pytorch as pl
import torch
import yaml
from chemprop.data import BatchMolGraph
from chemprop.nn import (
    Aggregation,
    MessagePassing,
    Predictor,
    MeanAggregation,
    BondMessagePassing,
)
from chemprop.nn.transforms import ScaleTransform
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ExponentialLR

from quarc.cli.train_args import TrainArgs
from quarc.data.gnn_dataloader import TrainingBatch_agent
from quarc.models.modules.gnn_heads import (
    GNNAgentAmountHead,
    GNNAgentHead,
    GNNAgentHeadWithRxnClass,
    GNNReactantAmountHead,
    GNNTemperatureHead,
)


class BaseGNN(pl.LightningModule):
    """Base GNN model for reaction prediction tasks, adapted from chemprop's MPNN."""

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["message_passing", "agg", "predictor"])
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        # {'metric_name': metric_function}, to use, metric_function(preds, targets)
        self.metrics = metrics

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self):
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """The learned fingerprints for the input molecules."""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def forward(
        self,
        agent_input: Tensor,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        graph_fp = self.fingerprint(bmg, V_d, X_d)
        return self.predictor(graph_fp, agent_input)

    def loss_fn(
        self,
        preds: Tensor,
        targets: Tensor,
        *args,
    ) -> Tensor:
        return self.criterion(preds, targets)

    def training_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        preds = self(a_input, bmg, V_d, X_d)
        l = self.loss_fn(preds, targets)
        self.log(
            "train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.predictor.output_transform.train()

    def validation_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(a_input, bmg, V_d, X_d)
        val_loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.init_lr)

        scheduler = OneCycleLR(
            opt,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_epoch_start(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_epoch=True, on_step=False, sync_dist=True)

    @classmethod
    def load_from_file_custom(cls, logger_dir: str, checkpoint_name: str, device: str = "cuda"):
        """Load a GNN model from a checkpoint and args.yaml file.

        Args:
            logger_dir: Directory containing the checkpoint and args.yaml files
            checkpoint_name: Name of the checkpoint file
            device: Device to load the model to ('cpu' or 'cuda')

        Returns:
            Loaded GNN model
        """
        PREDICTOR_MAP = {
            "AgentGNN": GNNAgentHead,
            "TemperatureGNN": GNNTemperatureHead,
            "ReactantAmountGNN": GNNReactantAmountHead,
            "AgentAmountOneshotGNN": GNNAgentAmountHead,
            "AgentGNNWithRxnClass": GNNAgentHeadWithRxnClass,
        }
        predictor_cls = PREDICTOR_MAP.get(cls.__name__)
        if predictor_cls is None:
            raise ValueError(f"Unsupported model class: {cls.__name__}")

        # Load checkpoint and args
        checkpoint_path = logger_dir + f"/{checkpoint_name}.ckpt"
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint.get("hyper_parameters", {})

        model_args_path = logger_dir + "/args.yaml"
        model_args = yaml.load(open(model_args_path, "r"), Loader=yaml.FullLoader)
        args = TrainArgs().from_dict(model_args)

        fdims = chemprop.featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF").shape
        message_passing = BondMessagePassing(*fdims, d_h=args.graph_hidden_size, depth=args.depth)
        predictor = predictor_cls(
            graph_input_dim=args.graph_hidden_size,
            agent_input_dim=args.num_classes,
            output_dim=args.output_size,
            hidden_dim=args.hidden_size,
            n_blocks=args.n_blocks,
        )
        aggregation = MeanAggregation()

        model = cls(
            message_passing=message_passing,
            agg=aggregation,
            predictor=predictor,
            batch_norm=True,
            metrics=hparams.get("metrics", None),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        return model


class AgentGNN(BaseGNN):
    def loss_fn(self, preds: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        """Sample-weighted cross entropy loss with mean reduction"""
        loss = self.criterion(preds, targets)
        return (loss * weights).mean()

    def training_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(a_input, bmg, V_d, X_d)
        l = self.loss_fn(preds, targets, weights)
        self.log(
            "train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(a_input, bmg, V_d, X_d)

        val_loss = self.loss_fn(preds, targets, weights)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.init_lr)
        lr_sched = ExponentialLR(optimizer=opt, gamma=0.98)
        return {"optimizer": opt, "lr_scheduler": lr_sched}


class AgentGNNWithRxnClass(BaseGNN):
    def forward(
        self,
        agent_input: Tensor,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        graph_fp = self.fingerprint(bmg)
        reaction_class_onehot = X_d
        return self.predictor(graph_fp, agent_input, reaction_class_onehot)

    def loss_fn(self, preds: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        """Sample-weighted cross entropy loss with mean reduction."""
        loss = self.criterion(preds, targets)
        return (loss * weights).mean()

    def training_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(a_input, bmg, V_d, X_d)
        l = self.loss_fn(preds, targets, weights)
        self.log(
            "train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(a_input, bmg, V_d, X_d)
        val_loss = self.loss_fn(preds, targets, weights)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.init_lr)
        lr_sched = ExponentialLR(optimizer=opt, gamma=0.98)
        return {"optimizer": opt, "lr_scheduler": lr_sched}


class TemperatureGNN(BaseGNN):
    """GNN model for predicting temperature."""

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        *args,
        **kwargs,
    ):
        if not isinstance(predictor, GNNTemperatureHead):
            raise TypeError("TemperatureGNN requires GNNTemperatureHead")
        super().__init__(message_passing, agg, predictor, *args, **kwargs)


class ReactantAmountGNN(BaseGNN):
    """GNN model for predicting binned reactant amounts."""

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        *args,
        **kwargs,
    ):
        if not isinstance(predictor, GNNReactantAmountHead):
            raise TypeError("ReactantAmountGNN requires GNNReactantAmountHead")

        super().__init__(message_passing, agg, predictor, *args, **kwargs)

    def forward(
        self,
        agent_input: Tensor,
        FP_reactants: Tensor,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        graph_fp = self.fingerprint(bmg, V_d, None)
        return self.predictor(graph_fp, agent_input, FP_reactants)

    def training_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, FP_reactants, targets, *_ = batch

        preds = self(a_input, FP_reactants, bmg, V_d)
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        l = self.loss_fn(preds, targets)
        self.log(
            "train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, FP_reactants, targets, *_ = batch

        preds = self(a_input, FP_reactants, bmg, V_d)
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(
                metric_name,
                metric,
                batch_size=len(batch[0]),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return val_loss


class AgentAmountOneshotGNN(BaseGNN):
    """GNN model for predicting binned agent amounts using one-shot approach."""

    def training_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, *_ = batch

        preds = self(a_input, bmg, V_d, X_d)
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        l = self.loss_fn(preds, targets)
        self.log(
            "train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch: TrainingBatch_agent, batch_idx):
        a_input, bmg, V_d, X_d, targets, *_ = batch

        preds = self(a_input, bmg, V_d, X_d)
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(
                metric_name,
                metric,
                batch_size=len(batch[0]),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return val_loss
