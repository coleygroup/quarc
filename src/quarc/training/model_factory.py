import os
import lightning.pytorch as pl
from torch.utils.data import DataLoader, DistributedSampler

from quarc.data.binning import BinningConfig
from quarc.settings import load as load_settings

cfg = load_settings()

class ModelFactory:
    """Factory for creating stage-specific models and datasets for ffn and gnn"""

    def __init__(self, args):
        self.args = args
        self._load_encoders()
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.args.binning_path:
            self.binning = BinningConfig.from_config(self.args.binning_path)
        else:
            self.binning = BinningConfig.default()

    def _load_encoders(self):
        """Load agent encoder and standardizer"""
        from quarc.models.modules.agent_encoder import AgentEncoder
        from quarc.models.modules.agent_standardizer import AgentStandardizer
        from quarc.models.modules.rxn_encoder import ReactionClassEncoder

        self.agent_encoder = AgentEncoder(
            class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
        )
        self.agent_standardizer = AgentStandardizer(
            conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
            other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
        )
        self.rxn_encoder = ReactionClassEncoder(
            class_path=cfg.get("pistachio_namerxn_path")
        )

    def create_model_and_data(
        self, train_data, val_data
    ) -> tuple[pl.LightningModule, DataLoader, DataLoader, list]:
        """Create model and data loaders for the specified stage"""

        if self.args.model_type == "ffn":
            return self._create_ffn_model_and_data(train_data, val_data)
        elif self.args.model_type == "gnn":
            return self._create_gnn_model_and_data(train_data, val_data)
        else:
            raise ValueError(f"Unknown model_type: {self.args.model_type}")

    def _create_ffn_model_and_data(self, train_data, val_data):
        """Create FFN model and datasets for current stage"""
        stage = self.args.stage

        if stage == 1:
            return self._create_ffn_stage1(train_data, val_data)
        elif stage == 2:
            return self._create_ffn_stage2(train_data, val_data)
        elif stage == 3:
            return self._create_ffn_stage3(train_data, val_data)
        elif stage == 4:
            return self._create_ffn_stage4(train_data, val_data)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _create_ffn_stage1(self, train_data, val_data):
        """FFN Stage 1: Agent prediction"""
        from quarc.data.ffn_datasets import AugmentedAgentsDatasetWithRxnClass, AgentsDatasetWithRxnClass
        from quarc.models.ffn_models import AgentFFNWithRxnClass
        from quarc.models.modules.ffn_heads import FFNAgentHeadWithRxnClass
        from quarc.models.callbacks import FFNGreedySearchCallback
        from torcheval.metrics.functional import multilabel_accuracy

        # Datasets
        train_dataset = AugmentedAgentsDatasetWithRxnClass(
            original_data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            rxn_encoder=self.rxn_encoder,
            sample_weighting="pascal",
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
        )
        val_dataset = AgentsDatasetWithRxnClass(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            rxn_encoder=self.rxn_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
        )

        # Data loaders
        if self.world_size > 1:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(train_dataset, shuffle=True),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(val_dataset, shuffle=False),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )

        # Model
        predictor = FFNAgentHeadWithRxnClass(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "multilabel_accuracy_exactmatch": multilabel_accuracy,
            "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
                preds, targets, criteria="hamming"
            ),
        }

        model = AgentFFNWithRxnClass(
            predictor=predictor,
            metrics=metrics,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        # Stage 1 specific callback
        greedy_callback = FFNGreedySearchCallback(track_batch_indices=range(10))
        extra_callbacks = [greedy_callback]

        return model, train_loader, val_loader, extra_callbacks

    def _create_ffn_stage2(self, train_data, val_data):
        """FFN Stage 2: Temperature prediction"""
        from quarc.data.ffn_datasets import BinnedTemperatureDataset
        from quarc.models.ffn_models import TemperatureFFN
        from quarc.models.modules.ffn_heads import FFNTemperatureHead
        from torchmetrics.classification import Accuracy

        # Datasets
        train_dataset = BinnedTemperatureDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.temperature_bins,
        )
        val_dataset = BinnedTemperatureDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.temperature_bins,
        )

        # Data loaders
        if self.world_size > 1:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(train_dataset, shuffle=True),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(val_dataset, shuffle=False),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )

        # Model
        if self.args.output_size != len(self.binning.temperature_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.temperature_bins) +1}.")

        predictor = FFNTemperatureHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = TemperatureFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_ffn_stage3(self, train_data, val_data):
        """FFN Stage 3: Reactant amount prediction"""
        from quarc.data.ffn_datasets import BinnedReactantAmountDataset
        from quarc.models.ffn_models import ReactantAmountFFN
        from quarc.models.modules.ffn_heads import FFNReactantAmountHead
        from torchmetrics.classification import Accuracy

        # Datasets
        train_dataset = BinnedReactantAmountDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.reactant_amount_bins,
        )
        val_dataset = BinnedReactantAmountDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.reactant_amount_bins,
        )

        # Data loaders
        if self.world_size > 1:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(train_dataset, shuffle=True),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(val_dataset, shuffle=False),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )

        # Model
        if self.args.output_size != len(self.binning.reactant_amount_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.reactant_amount_bins) +1}.")

        predictor = FFNReactantAmountHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = ReactantAmountFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_ffn_stage4(self, train_data, val_data):
        """FFN Stage 4: Agent amount prediction"""
        from quarc.data.ffn_datasets import BinnedAgentAmoutOneshot
        from quarc.models.ffn_models import AgentAmountFFN
        from quarc.models.modules.ffn_heads import FFNAgentAmountHead
        from torchmetrics.classification import Accuracy

        # Datasets
        train_dataset = BinnedAgentAmoutOneshot(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.agent_amount_bins,
        )
        val_dataset = BinnedAgentAmoutOneshot(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            fp_radius=self.args.fp_radius,
            fp_length=self.args.fp_length,
            bins=self.binning.agent_amount_bins,
        )

        # Data loaders
        if self.world_size > 1:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(train_dataset, shuffle=True),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                persistent_workers=False,
                pin_memory=True,
                sampler=DistributedSampler(val_dataset, shuffle=False),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )

        # Model
        if self.args.output_size != len(self.binning.agent_amount_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.agent_amount_bins) +1}.")

        predictor = FFNAgentAmountHead(
            fp_dim=self.args.fp_length,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = AgentAmountFFN(
            predictor=predictor,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_model_and_data(self, train_data, val_data):
        """Create GNN model and datasets for current stage"""
        stage = self.args.stage

        if stage == 1:
            return self._create_gnn_stage1(train_data, val_data)
        elif stage == 2:
            return self._create_gnn_stage2(train_data, val_data)
        elif stage == 3:
            return self._create_gnn_stage3(train_data, val_data)
        elif stage == 4:
            return self._create_gnn_stage4(train_data, val_data)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _create_gnn_stage1(self, train_data, val_data):
        """GNN Stage 1: Agent prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNAugmentedAgentsDatasetWithRxnClass, GNNAgentsDatasetWithRxnClass
        from quarc.models.gnn_models import AgentGNNWithRxnClass
        from quarc.models.modules.gnn_heads import GNNAgentHeadWithRxnClass
        from quarc.models.callbacks import GNNGreedySearchCallback
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torcheval.metrics.functional import multilabel_accuracy

        # Featurizer and reaction encoder
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNAugmentedAgentsDatasetWithRxnClass(
            original_data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            rxn_encoder=self.rxn_encoder,
            featurizer=featurizer,
        )
        val_dataset = GNNAgentsDatasetWithRxnClass(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            rxn_encoder=self.rxn_encoder,
            featurizer=featurizer,
        )

        # Data loaders
        from quarc.data.gnn_dataloader import build_dataloader_agent


        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            distributed=self.world_size > 1,
            persistent_workers=False,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            distributed=self.world_size > 1,
            persistent_workers=False,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNAgentHeadWithRxnClass(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=len(self.agent_encoder),
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "multilabel_accuracy_exactmatch": multilabel_accuracy,
            "multilabel_accuracy_hamming": lambda preds, targets: multilabel_accuracy(
                preds, targets, criteria="hamming"
            ),
        }

        model = AgentGNNWithRxnClass(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            init_lr=self.args.init_lr,
        )

        # Stage 1 specific callback
        greedy_callback = GNNGreedySearchCallback(track_batch_indices=range(len(val_loader)))
        extra_callbacks = [greedy_callback]

        return model, train_loader, val_loader, extra_callbacks

    def _create_gnn_stage2(self, train_data, val_data):
        """GNN Stage 2: Temperature prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedTemperatureDataset
        from quarc.models.gnn_models import TemperatureGNN
        from quarc.models.modules.gnn_heads import GNNTemperatureHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNBinnedTemperatureDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.temperature_bins,
        )
        val_dataset = GNNBinnedTemperatureDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.temperature_bins,
        )

        # Data loaders
        from quarc.data.gnn_dataloader import build_dataloader_agent

        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNTemperatureHead(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }
        if self.args.output_size != len(self.binning.temperature_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.temperature_bins) +1}.")

        model = TemperatureGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_stage3(self, train_data, val_data):
        """GNN Stage 3: Reactant amount prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedReactantAmountDataset
        from quarc.models.gnn_models import ReactantAmountGNN
        from quarc.models.modules.gnn_heads import GNNReactantAmountHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNBinnedReactantAmountDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.reactant_amount_bins,
        )
        val_dataset = GNNBinnedReactantAmountDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.reactant_amount_bins,
        )

        # Data loaders
        from quarc.data.gnn_dataloader import build_dataloader_agent

        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNReactantAmountHead(
            graph_input_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        if self.args.output_size != len(self.binning.reactant_amount_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.reactant_amount_bins) +1}.")

        model = ReactantAmountGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

    def _create_gnn_stage4(self, train_data, val_data):
        """GNN Stage 4: Agent amount prediction"""
        import chemprop
        from chemprop import featurizers
        from quarc.data.gnn_datasets import GNNBinnedAgentAmountOneShotDataset
        from quarc.models.gnn_models import AgentAmountOneshotGNN
        from quarc.models.modules.gnn_heads import GNNAgentAmountHead
        from quarc.data.gnn_dataloader import build_dataloader_agent
        from torchmetrics.classification import Accuracy

        # Featurizer
        featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

        # Datasets
        train_dataset = GNNBinnedAgentAmountOneShotDataset(
            data=train_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.agent_amount_bins,
        )
        val_dataset = GNNBinnedAgentAmountOneShotDataset(
            data=val_data,
            agent_standardizer=self.agent_standardizer,
            agent_encoder=self.agent_encoder,
            featurizer=featurizer,
            bins=self.binning.agent_amount_bins,
        )

        from quarc.data.gnn_dataloader import build_dataloader_agent

        train_loader = build_dataloader_agent(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )
        val_loader = build_dataloader_agent(
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            classification=True,
            distributed=self.world_size > 1,
            persistent_workers=True,
            pin_memory=True,
        )

        # Model
        fdims = featurizer.shape
        mp = chemprop.nn.BondMessagePassing(
            *fdims, d_h=self.args.graph_hidden_size, depth=self.args.depth
        )
        agg = chemprop.nn.MeanAggregation()

        predictor = GNNAgentAmountHead(
            graph_dim=self.args.graph_hidden_size,
            agent_input_dim=len(self.agent_encoder),
            output_dim=self.args.output_size,
            hidden_dim=self.args.hidden_size,
            n_blocks=self.args.n_blocks,
        )

        if self.args.output_size != len(self.binning.agent_amount_bins) + 1:
            raise ValueError(f"output_size {self.args.output_size} does not match binning. Expected to be {len(self.binning.agent_amount_bins) +1}.")

        metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                average="micro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
            "accuracy_macro": Accuracy(
                task="multiclass",
                average="macro",
                num_classes=self.args.output_size,
                ignore_index=0,
            ),
        }

        model = AgentAmountOneshotGNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            batch_norm=True,
            metrics=metrics,
            warmup_epochs=self.args.warmup_epochs,
            init_lr=self.args.init_lr,
            max_lr=self.args.max_lr,
            final_lr=self.args.final_lr,
        )

        return model, train_loader, val_loader, []

