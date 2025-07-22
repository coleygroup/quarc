import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Iterator

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.data.binning import BinningConfig
from quarc.data.gnn_datasets import (
    rxn_smiles_to_mols,
    standardize_and_encode_agents,
    MAX_NUM_REACTANTS,
)
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.utils.smiles_utils import prep_rxn_smi_input


@dataclass
class ReactionInput:
    """Clean dataclass for evaluation items"""

    metadata: dict[str, Any]
    model_inputs: dict[str, Any]
    targets: dict[str, Any]
    raw_data: Optional[ReactionDatum] = None

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")

    @property
    def rxn_smiles(self) -> str:
        return self.metadata.get("rxn_smiles", "")


class UnifiedEvaluationDataset:
    """
    Dataset for unified evaluation of all stages, prepare inputs for all stages.

    The choices of everything but data should be consistent with the trained model.
    Allows overriding agents for prediction. Just prepares the tensor for entered agents.

    Metadata:
        rxn_smiles: The SMILES representation of the reaction.
        rxn_class: The class of the reaction.
        document_id: The unique identifier for the document.

    Model Inputs:
        mg: The molecular graph representation.
        FP_inputs: Fingerprints for feedforward neural network models.
        FP_reactants: Fingerprints of reactants, padded for reactant amount prediction.
        rxn_class: One-hot encoded reaction class.

    Targets:
        target_agents: List of target agents.
        target_temp: Target temperature.
        target_reactant_amounts: List of target reactant amounts, padded up to MAX_NUM_REACTANTS.
        target_agent_amounts: List of target agent amounts, padded up to the length of the agent encoder.

    """

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        rxn_encoder: ReactionClassEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        binning_config: BinningConfig = None,
        include_raw_data: bool = False,
        include_model_inputs: bool = True,
        include_targets: bool = True,
        fp_radius: int = 3,
        fp_length: int = 2048,
    ):
        self.data = data
        self.include_raw_data = include_raw_data
        self.include_model_inputs = include_model_inputs
        self.include_targets = include_targets

        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.featurizer = featurizer
        self.rxn_class_encoder = rxn_encoder

        self.binning_config = binning_config or BinningConfig.default()

        self.fp_radius = fp_radius
        self.fp_length = fp_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> ReactionInput:
        """Gets metadata, model inputs, and targets for a given reaction.

        Args:
            idx: The index of the reaction to get.
        Returns:
            ReactionInput: A dataclass containing the metadata, model inputs, and targets.
        """

        # try:
        #     targets = self._get_targets(self.data[idx]) if self.include_targets else {}
        # except TypeError as e:
        #     print(f"Error getting targets{e}")
        #     targets = {
        #         "target_agents": np.array([0]),
        #         "target_temp": 0,
        #         "target_reactant_amounts": np.array([0]),
        #         "target_agent_amounts": [(0, 0)],
        #     }

        return ReactionInput(
            metadata=self._get_metadata(self.data[idx]),
            model_inputs=(
                self._get_model_inputs(self.data[idx]) if self.include_model_inputs else {}
            ),
            targets=self._get_targets(self.data[idx]) if self.include_targets else {},
            raw_data=self.data[idx] if self.include_raw_data else None,
        )

    def __iter__(self) -> Iterator[ReactionInput]:
        for idx in range(len(self)):
            yield self[idx]

    def _get_metadata(self, datum: ReactionDatum) -> dict:
        return {
            "rxn_smiles": datum.rxn_smiles,
            "rxn_class": datum.rxn_class,
            "doc_id": datum.document_id,
        }

    def _get_model_inputs(self, datum: ReactionDatum) -> dict:
        rxn_smiles = prep_rxn_smi_input(datum.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)
        rxn_class = datum.rxn_class

        if rxn_mols[0] is None or rxn_mols[1] is None:
            raise ValueError(f"Invalid reaction: {datum.document_id}, need dummy mg")

        mg = self.featurizer(rxn_mols)

        # Generate fingerprints for FFN models
        FP_inputs = self.generate_reaction_fingerprint(datum)

        # for reactant amount prediction, pad reactants to MAX_NUM_REACTANTS
        FP_reactants = np.stack(
            [
                self.generate_fingerprint([reactant])
                for reactant in datum.reactants[:MAX_NUM_REACTANTS]
            ]
        )
        if FP_reactants.shape[0] < MAX_NUM_REACTANTS:
            FP_reactants = np.pad(
                FP_reactants,
                ((0, MAX_NUM_REACTANTS - len(FP_reactants)), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        return {
            "mg": mg,
            "FP_inputs": FP_inputs,
            "FP_reactants": FP_reactants,
            "rxn_class": self.rxn_class_encoder.to_onehot(rxn_class),
        }

    def _get_targets(self, datum: ReactionDatum) -> dict:
        # get target agents
        a_idxs = standardize_and_encode_agents(
            datum.agents, self.agent_standardizer, self.agent_encoder
        )
        target_agents = a_idxs.tolist()

        # get target temperature
        if datum.temperature is None:
            target_temp = 0
        else:
            target_temp = np.digitize(datum.temperature, self.binning_config.temperature_bins)

        # get target reactant amounts
        limiting_reactant_amount = min([reactant.amount for reactant in datum.reactants])
        target_reactant_amounts = [
            reactant.amount / limiting_reactant_amount
            for reactant in datum.reactants[:MAX_NUM_REACTANTS]
        ]
        target_reactant_amounts = np.digitize(
            target_reactant_amounts, self.binning_config.reactant_amount_bins
        ).tolist()

        # get target agent amounts
        target_agent_amounts = []  # (a_idx, bin_idx)
        for i, agent in enumerate(datum.agents):
            standardized_smiles = self.agent_standardizer.standardize([agent.smiles])
            agent_idx = self.agent_encoder.encode(standardized_smiles)[0]
            relative_amount = agent.amount / limiting_reactant_amount
            target_agent_amounts.append(
                (
                    agent_idx,
                    np.digitize([relative_amount], self.binning_config.agent_amount_bins)[0],
                )
            )

        return {
            "target_agents": target_agents,
            "target_temp": target_temp,
            "target_reactant_amounts": target_reactant_amounts,
            "target_agent_amounts": target_agent_amounts,
        }

    def get_bin_labels(self, bin_type="agent"):
        bin_labels = {}
        bins = None

        if bin_type == "temperature":
            bins = self.binning_config.temperature_bins
        elif bin_type == "reactant":
            bins = self.binning_config.reactant_amount_bins
        elif bin_type == "agent":
            bins = self.binning_config.agent_amount_bins
        else:
            raise ValueError(f"Unknown bin_type: {bin_type}")

        for i in range(len(bins) + 1):
            if i == 0:
                label = f"(-inf, {bins[i]:.2f})"
            elif i == len(bins):
                label = f"[{bins[i-1]:.2f}, inf)"
            else:
                label = f"[{bins[i-1]:.2f}, {bins[i]:.2f})"
            bin_labels[i] = label
        return bin_labels

    def generate_fingerprint(self, agent_records: list[AgentRecord]) -> np.ndarray:
        smi_list = [agent_record.smiles for agent_record in agent_records]
        merged_smi = ".".join(smi_list)
        mol = Chem.MolFromSmiles(merged_smi)

        fp_arr = np.zeros((2048,), dtype=bool)
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            fp_arr = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_length
            )
        return torch.tensor(fp_arr, dtype=torch.bool)

    def generate_reaction_fingerprint(self, reaction_datum: ReactionDatum) -> torch.Tensor:
        FP_r = self.generate_fingerprint(reaction_datum.reactants)
        FP_p = self.generate_fingerprint(reaction_datum.products)
        return torch.cat((FP_r, FP_p))


class EvaluationDatasetFactory:
    """Dataset factory for creating evaluation datasets."""

    @staticmethod
    def for_models(
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        rxn_encoder: ReactionClassEncoder,
        **kwargs,
    ) -> UnifiedEvaluationDataset:
        """Create dataset for model (with model inputs)."""
        return UnifiedEvaluationDataset(
            data=data,
            agent_standardizer=agent_standardizer,
            agent_encoder=agent_encoder,
            featurizer=featurizer,
            rxn_encoder=rxn_encoder,
            include_raw_data=False,
            include_model_inputs=True,
            **kwargs,
        )

    @staticmethod
    def for_baseline_with_targets(
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        rxn_encoder: ReactionClassEncoder,
        **kwargs,
    ) -> UnifiedEvaluationDataset:
        """Create dataset without model inputs but has targets."""
        return UnifiedEvaluationDataset(
            data=data,
            agent_standardizer=agent_standardizer,
            agent_encoder=agent_encoder,
            featurizer=featurizer,
            rxn_encoder=rxn_encoder,
            include_raw_data=False,
            include_model_inputs=False,
            **kwargs,
        )

    @staticmethod
    def for_baseline(data: list[ReactionDatum]) -> list[ReactionInput]:
        """Create dataset with meta data only."""
        return [
            ReactionInput(
                metadata={
                    "doc_id": datum.document_id,
                    "rxn_class": datum.rxn_class,
                    "rxn_smiles": datum.rxn_smiles,
                },
                model_inputs={},
                targets={},
                raw_data=datum,
            )
            for datum in data
        ]

    @staticmethod
    def for_inference(
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        rxn_encoder: ReactionClassEncoder,
        **kwargs,
    ) -> UnifiedEvaluationDataset:
        """Create dataset for model (with model inputs)."""
        return UnifiedEvaluationDataset(
            data=data,
            agent_standardizer=agent_standardizer,
            agent_encoder=agent_encoder,
            featurizer=featurizer,
            rxn_encoder=rxn_encoder,
            include_raw_data=True,
            include_model_inputs=True,
            include_targets=False,
            **kwargs,
        )
