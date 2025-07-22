import math
import random
from itertools import combinations
from typing import Iterator, Literal

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.settings import load as load_settings

cfg = load_settings()

RDLogger.DisableLog("rdApp.*")

MAX_NUM_AGENTS = 5
MAX_NUM_REACTANTS = 5


class ReactionDatasetBase(Dataset):
    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
    ):

        self.data = data
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.fp_radius = fp_radius
        self.fp_length = fp_length

    def generate_fingerprint(self, agent_records: list[AgentRecord]) -> torch.Tensor:
        """Generate fingerprints for given reactants or products (AgentRecords) of one rxn, and aggregate into a single combined 2048-bit vector.

        Returns:
            combined 2048-bit boolean tensor for input reactants or products AgentRecords

        Examples:
            >>> F_r = generate_fingerprint(reactants)
            >>> F_p = generate_fingerprint(products)
        """

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

    def standardize_and_encode_agents(self, agents: list[AgentRecord]) -> torch.Tensor:
        """Standardize and encode agents to a tensor of indices and scatter to return a multi-hot tensor."""

        agent_smiles = [agent.smiles for agent in agents]
        standardized_smiles = self.agent_standardizer.standardize(agent_smiles)
        a_idxs = torch.tensor(self.agent_encoder.encode(standardized_smiles))
        return torch.zeros(len(self.agent_encoder)).scatter(-1, a_idxs, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass


class AugmentedAgentsDataset(Dataset):
    """
    Dataset for generating all possible combinations of splitting to input and target agents.

    Uses precomputed index mapping for efficient access while maintaining lazy computation of features.
    """

    def __init__(
        self,
        original_data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        rxn_encoder: ReactionClassEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
        sample_weighting: Literal["pascal", "uniform", "none"] = "pascal",
    ):
        super().__init__()
        self.original_data = original_data
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.rxn_encoder = rxn_encoder
        self.fp_radius = fp_radius
        self.fp_length = fp_length
        self.sample_weighting = sample_weighting

        self.index_mapping = self._create_index_mapping()

    def _create_index_mapping(self):
        """Create mapping between original index and augmented index.

        Either as (original_idx, -1) for base case (full set) or (original_idx, (r, comb_idx)) for combinations.
        """

        mapping = []
        for orig_idx, datum in enumerate(self.original_data):
            agent_smiles = [agent.smiles for agent in datum.agents]
            standardized_smiles = self.agent_standardizer.standardize(agent_smiles)
            a_idxs_orig = self.agent_encoder.encode(standardized_smiles)
            # base case (full set)
            mapping.append((orig_idx, -1))

            # combinations
            for r in range(1, len(a_idxs_orig) + 1):
                for combo_idx, _ in enumerate(combinations(range(len(a_idxs_orig)), r)):
                    mapping.append((orig_idx, (r, combo_idx)))
        return mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, float, Tensor]:
        """Get item from index mapping.

        Args:
            idx: dataset index.

        Returns:
            FP_input: reaction fingerprint
            a_input: input agents as multi-hot tensor
            a_target: target agents as multi-hot tensor
            weight: weight of the training sample
            rxn_class: reaction class as one-hot tensor
        """

        orig_idx, aug_spec = self.index_mapping[idx]
        original_datum = self.original_data[orig_idx]

        # Generate reaction fingerprint
        FP_input = self.generate_reaction_fingerprint(original_datum)
        rxn_class = torch.tensor(self.rxn_encoder.to_onehot(original_datum.rxn_class))

        # Get agent indices
        agent_smiles = [agent.smiles for agent in original_datum.agents]
        standardized_smiles = self.agent_standardizer.standardize(agent_smiles)
        a_idxs_orig = torch.tensor(self.agent_encoder.encode(standardized_smiles))
        a_orig = torch.zeros(len(self.agent_encoder)).scatter(-1, a_idxs_orig, 1)

        # Calculate weights
        if self.sample_weighting == "pascal":
            all_sample_weights = self.pascal_triangle_weights(len(a_idxs_orig))
        elif self.sample_weighting == "uniform":
            all_sample_weights = torch.full((2 ** len(a_idxs_orig),), 1 / (2 ** len(a_idxs_orig)))
        elif self.sample_weighting == "none":
            all_sample_weights = torch.full((2 ** len(a_idxs_orig),), 1)
        else:
            raise ValueError(f"Invalid sample_weighting: {self.sample_weighting}")

        # Handle base case vs combinations
        if aug_spec == -1:
            # Full set of agents as input, EOS token as target
            a_input = a_orig
            a_target = torch.zeros_like(a_orig).scatter(-1, torch.tensor([0]), 1)
            weight = all_sample_weights[0]
        else:
            r, combo_idx = aug_spec
            all_combos = list(combinations(range(len(a_idxs_orig)), r))
            combo = all_combos[combo_idx]
            a_input = a_orig.clone()
            a_input[a_idxs_orig[list(combo)]] = 0
            a_target = a_orig - a_input
            weight = all_sample_weights[r]

        return FP_input, a_input, a_target, weight, rxn_class

    def pascal_triangle_weights(self, n_agents: int) -> torch.Tensor:
        """Generates weights based on Pascal's Triangle for a given number of agents."""
        weights = [math.comb(n_agents, k) for k in range(n_agents + 1)]
        inv_weights = [1 / w for w in weights]
        return torch.tensor(inv_weights)

    def generate_fingerprint(self, agent_records: list[AgentRecord]) -> torch.Tensor:
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


class AgentsDatasetWithReactionClass(ReactionDatasetBase):
    """Regular Agent Dataset for validation and testing with reaction class.

    For each reaction, the input is the FP and no agent, and the target is all agents.
    """

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        rxn_encoder: ReactionClassEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
    ):
        """Initialize the AgentsDatasetWithReactionClass.

        Args:
            data: list of ReactionDatum
            agent_standardizer: AgentStandardizer
            agent_encoder: AgentEncoder
            fp_radius: default 3
            fp_length: default 2048
        """

        super().__init__(data, agent_standardizer, agent_encoder, fp_radius, fp_length)
        self.rxn_encoder = rxn_encoder

    def __getitem__(self, idx):
        """Get item from index mapping.

        Args:
            idx: dataset index.

        Returns:
            FP_input: reaction fingerprint
            a_input: empty multi-hot tensor
            a_target: multi-hot tensor of all agents
            rxn_class: reaction class as one-hot tensor
        """

        datum = self.data[idx]

        FP_input = self.generate_reaction_fingerprint(datum)
        rxn_class = torch.tensor(self.rxn_encoder.to_onehot(datum.rxn_class))

        a_target = self.standardize_and_encode_agents(datum.agents)
        a_input = torch.zeros_like(a_target)

        return FP_input, a_input, a_target, rxn_class

    def _getname(self):
        return "AgentsDataset"


class BinnedTemperatureDataset(ReactionDatasetBase):
    """Dataset for temperature prediction."""

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
        bins: None | list[float] = None,
    ):
        """Initialize the BinnedTemperatureDataset.

        Args:
            data: list of ReactionDatum
            agent_standardizer: AgentStandardizer
            agent_encoder: AgentEncoder
            fp_radius: default 3
            fp_length: default 2048
            bins: temperature bin edges, default to use 10-degree intervals from -100 to 200 C
        """

        super().__init__(data, agent_standardizer, agent_encoder, fp_radius, fp_length)
        if bins is None:
            self.bins = np.arange(-100, 201, 10) + 273.15
        else:
            self.bins = bins

    @property
    def bin_labels(self):
        bins_in_celsius = [bin - 273.15 for bin in self.bins]

        bin_labels = {}
        for i in range(len(bins_in_celsius) + 1):
            if i == 0:
                label = f"(-inf, {bins_in_celsius[i]:.2f})"
            elif i == len(bins_in_celsius):
                label = f"[{bins_in_celsius[i-1]:.2f}, inf)"
            else:
                label = f"[{bins_in_celsius[i-1]}, {bins_in_celsius[i]})"
            bin_labels[i] = label
        return bin_labels

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        """Get item from index mapping.

        Args:
            idx: dataset index.

        Returns:
            FP_input: reaction fingerprint
            a_input: ground truth input agents as multi-hot tensor
            T_target: target temperature as index
        """

        datum = self.data[idx]

        FP_input = self.generate_reaction_fingerprint(datum)

        a_input = self.standardize_and_encode_agents(datum.agents)

        T_target = torch.tensor(np.digitize(datum.temperature, self.bins)).long()
        return FP_input, a_input, T_target


class BinnedReactantAmountDataset(ReactionDatasetBase):
    """Dataset for reactant amount prediction."""

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
        bins: None | list[float] = None,
    ):
        """Initialize the BinnedReactantAmountDataset.

        Args:
            data: list of ReactionDatum
            agent_standardizer: AgentStandardizer
            agent_encoder: AgentEncoder
            fp_radius: default 3
            fp_length: default 2048
            bins: reactant amount bin edges, default to use 0.05-step intervals from 0.95 to 100.5
        """

        super().__init__(data, agent_standardizer, agent_encoder, fp_radius, fp_length)
        if bins is None:
            self.bins = [
                0.95,
                1.05,
                1.15,
                1.25,
                1.35,
                1.45,
                1.75,
                2.25,
                2.75,
                3.5,
                4.5,
                5.5,
                6.5,
                7.5,
            ]
        else:
            self.bins = bins

    @property
    def bin_labels(self):
        bin_labels = {}
        for i in range(len(self.bins) + 1):
            if i == 0:
                label = f"(-inf, {self.bins[i]:.2f})"
            elif i == len(self.bins):
                label = f"[{self.bins[i-1]:.2f}, inf)"
            else:
                label = f"[{self.bins[i-1]}, {self.bins[i]})"
            bin_labels[i] = label
        return bin_labels

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get item from index mapping. Padded to MAX_NUM_REACTANTS.

        Args:
            idx: dataset index.

        Returns:
            FP_input: reaction fingerprint (b, 4096)
            a_input: ground truth input agents as multi-hot tensor (b, num_classes)
            FP_reactants: reactants as multi-hot tensor (b, 5, 2048)
            r_amount_target: target reactant amount as index (b, 5)
        """

        datapoint = self.data[idx]

        FP_input = self.generate_reaction_fingerprint(datapoint)
        a_input = self.standardize_and_encode_agents(datapoint.agents)
        FP_reactants = torch.stack(
            [
                self.generate_fingerprint([reactant])
                for reactant in datapoint.reactants[:MAX_NUM_REACTANTS]
            ]
        ).float()

        limiting_reactant_amount = min([reactant.amount for reactant in datapoint.reactants])
        r_amount = [
            reactant.amount / limiting_reactant_amount
            for reactant in datapoint.reactants[:MAX_NUM_REACTANTS]
        ]
        r_amount_target = torch.tensor(np.digitize(r_amount, self.bins))

        # pad to MAX_NUM_REACTANTS with 0, invalid reactants has target class of 0 (invalid according to bin_labels)
        if len(datapoint.reactants) < MAX_NUM_REACTANTS:
            FP_reactants = torch.cat(
                (
                    FP_reactants,
                    torch.zeros(
                        MAX_NUM_REACTANTS - len(datapoint.reactants), 2048, dtype=torch.bool
                    ),
                )
            )
            r_amount_target = torch.cat(
                (r_amount_target, torch.zeros(MAX_NUM_REACTANTS - len(datapoint.reactants)))
            )

        return FP_input, a_input, FP_reactants, r_amount_target.long()


class BinnedAgentAmoutOneshot(ReactionDatasetBase):
    """Dataset for Agent amount prediction."""

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        fp_radius: int = 3,
        fp_length: int = 2048,
        bins: None | list[float] = None,
    ):
        """Initialize the BinnedAgentAmoutOneshot.

        Args:
            data: list of ReactionDatum
            agent_standardizer: AgentStandardizer
            agent_encoder: AgentEncoder
            fp_radius: default 3
            fp_length: default 2048
            bins: agent amount bin edges
        """

        super().__init__(data, agent_standardizer, agent_encoder, fp_radius, fp_length)
        if bins is None:
            small_bins = np.array([0, 0.075, 0.15, 0.25, 0.55, 0.95])
            regular_bins = np.array([1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            large_bins = np.array([15.5, 25.5, 35.5, 45.5, 55.5, 65.5, 75.5, 85.5, 100.5])
            self.bins = np.concatenate([small_bins, regular_bins, large_bins])
        else:
            self.bins = bins

    @property
    def bin_labels(self):
        bin_labels = {}
        for i in range(len(self.bins) + 1):
            if i == 0:
                label = f"(-inf, {self.bins[i]:.2f})"
            elif i == len(self.bins):
                label = f"[{self.bins[i-1]:.2f}, inf)"
            else:
                label = f"[{self.bins[i-1]}, {self.bins[i]})"
            bin_labels[i] = label
        return bin_labels

    def __getitem__(self, idx):
        """Get item from index mapping.

        Args:
            idx: dataset index.

        Returns:
            FP_input: reaction fingerprint (b, 4096)
            a_input: ground truth input agents as multi-hot tensor (b, num_classes)
            a_amount_target: target agent amount as index (b, num_classes)
        """

        datapoint = self.data[idx]

        FP_input = self.generate_reaction_fingerprint(datapoint)
        a_input = self.standardize_and_encode_agents(datapoint.agents)

        limiting_reactant_amount = min([reactant.amount for reactant in datapoint.reactants])
        a_amount_target = torch.zeros(len(a_input), dtype=torch.long)
        for _i, agent in enumerate(datapoint.agents[:MAX_NUM_AGENTS]):
            standardized_smiles = self.agent_standardizer.standardize([agent.smiles])
            agent_idx = self.agent_encoder.encode(standardized_smiles)[0]
            relative_amount = agent.amount / limiting_reactant_amount
            a_amount_target[agent_idx] = torch.tensor(np.digitize([relative_amount], self.bins)[0])

        return FP_input, a_input, a_amount_target.long()
