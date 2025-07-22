from typing import Callable

import lightning.pytorch as pl
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR

from quarc.cli.train_args import TrainArgs
from quarc.models.modules.ffn_heads import (
    FFNBaseHead,
    FFNAgentAmountHead,
    FFNAgentHeadWithReactionClass,
    FFNAgentHead,
    FFNReactantAmountHead,
    FFNTemperatureHead,
)


class BaseFFN(pl.LightningModule):
    """Base FFN model for reaction prediction tasks."""

    def __init__(
        self,
        predictor: FFNBaseHead,
        metrics: dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()
        if metrics is None:
            raise ValueError("Need callable metrics")
        self.save_hyperparameters(ignore=["predictor", "metrics"])

        self.predictor = predictor
        self.criterion = predictor.criterion
        self.metrics = metrics

        # Learning rate parameters
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    def forward(self, FP_inputs: Tensor, agent_input: Tensor) -> Tensor:
        return self.predictor(FP_inputs, agent_input)

    def loss_fn(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return self.criterion(preds, targets)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

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

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

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

    def on_train_epoch_start(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_epoch=True, on_step=False, sync_dist=True)

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

    @classmethod
    def load_from_file_custom(cls, logger_dir: str, checkpoint_name: str, device: str = "cuda"):
        """Load a FFN model from a checkpoint and args.yaml file."""
        PREDICTOR_MAP = {
            "AgentFFN": FFNAgentHead,
            "TemperatureFFN": FFNTemperatureHead,
            "ReactantAmountFFN": FFNReactantAmountHead,
            "AgentAmountFFN": FFNAgentAmountHead,
            "AgentFFNWithReactionClass": FFNAgentHeadWithReactionClass,
        }
        predictor_cls = PREDICTOR_MAP.get(cls.__name__)
        if predictor_cls is None:
            raise ValueError(f"Unsupported model class: {cls.__name__}")

        # Load checkpoint and args
        checkpoint = torch.load(f"{logger_dir}/{checkpoint_name}.ckpt")
        hparams = checkpoint.get("hyper_parameters", {})

        model_args_path = logger_dir + "/args.yaml"
        model_args = yaml.load(open(model_args_path, "r"), Loader=yaml.FullLoader)
        args = TrainArgs().from_dict(model_args)

        # Initialize predictor
        predictor = predictor_cls(
            fp_dim=args.FP_length,
            agent_input_dim=args.num_classes,
            output_dim=args.output_size,
            hidden_dim=args.hidden_size,
            n_blocks=args.n_blocks,
            activation=args.activation,
        )

        # Initialize and load model
        model = cls(
            predictor=predictor,
            metrics=hparams.get("metrics", []),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        return model


class AgentFFNWithReactionClass(BaseFFN):
    """FFN model for agent prediction with sample weights"""

    def __init__(
        self,
        predictor: FFNAgentHeadWithReactionClass,
        *args,
        **kwargs,
    ):
        if not isinstance(predictor, FFNAgentHeadWithReactionClass):
            raise TypeError("AgentFFN requires FFNAgentHeadWithReactionClass")
        super().__init__(predictor, *args, **kwargs)

    def loss_fn(self, preds: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        loss = self.criterion(preds, targets)
        return (loss * weights).mean()

    def forward(self, FP_inputs: Tensor, agent_input: Tensor, rxn_class: Tensor) -> Tensor:
        return self.predictor(FP_inputs, agent_input, rxn_class)

    def training_step(self, batch, batch_idx):

        FP_inputs, a_inputs, targets, weights, rxn_class = batch
        preds = self(FP_inputs, a_inputs, rxn_class)
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

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets, rxn_class = batch
        preds = self(FP_inputs, a_inputs, rxn_class)
        dummy_weights = torch.ones(FP_inputs.shape[0]).to(preds.device)

        val_loss = self.loss_fn(preds, targets, dummy_weights)
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
        """For training, count total steps for augmented training data"""
        opt = Adam(self.parameters(), lr=self.init_lr)

        lr_sched = ExponentialLR(optimizer=opt, gamma=0.98)

        return {"optimizer": opt, "lr_scheduler": lr_sched}


class TemperatureFFN(BaseFFN):
    """FFN model for predicting temperature."""

    def __init__(self, predictor: FFNTemperatureHead, *args, **kwargs):
        if not isinstance(predictor, FFNTemperatureHead):
            raise TypeError("TemperatureFFN requires FFNTemperatureHead")
        super().__init__(predictor, *args, **kwargs)


class ReactantAmountFFN(BaseFFN):
    """FFN model for predicting binned reactant amounts."""

    def __init__(self, predictor: FFNReactantAmountHead, *args, **kwargs):
        if not isinstance(predictor, FFNReactantAmountHead):
            raise TypeError("ReactantAmountFFN requires FFNReactantAmountHead")
        super().__init__(predictor, *args, **kwargs)

    def forward(
        self,
        FP_inputs: Tensor,
        agent_input: Tensor,
        FP_reactants: Tensor,
    ) -> Tensor:
        return self.predictor(FP_inputs, agent_input, FP_reactants)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, FP_reactants, targets = batch
        preds = self(
            FP_inputs, a_inputs, FP_reactants
        )  # (batch_size, MAX_NUM_REACTANTS, num_binned_classes)
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

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, FP_reactants, targets = batch
        preds = self(
            FP_inputs, a_inputs, FP_reactants
        )  # (batch_size, MAX_NUM_REACTANTS, num_binned_classes)
        preds = preds.view(
            -1, preds.shape[-1]
        )  # (batch_size * MAX_NUM_REACTANTS, num_binned_classes)
        targets = targets.view(-1)  # (batch_size * MAX_NUM_REACTANTS)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss


class AgentAmountFFN(BaseFFN):
    """FFN model for predicting binned agent amounts"""

    def __init__(self, predictor: FFNAgentAmountHead, *args, **kwargs):
        if not isinstance(predictor, FFNAgentAmountHead):
            raise TypeError("AgentAmountFFN requires FFNAgentAmountHead")
        super().__init__(predictor, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)  # (batch_size, num_classes, num_bins)

        # flatten preds and targets
        preds = preds.view(-1, preds.shape[-1])  # (batch_size * num_classes, num_bins)
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

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

        # flatten preds and targets
        preds = preds.view(-1, preds.shape[-1])  # (batch_size * num_classes, num_bins)
        targets = targets.view(-1)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss",
            val_loss,
            batch_size=len(batch[0]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss
