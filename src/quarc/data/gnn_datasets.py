import math
from typing import NamedTuple
from itertools import combinations
from loguru import logger
import torch
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolDescriptors
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer
from chemprop.utils import make_mol

from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.utils.smiles_utils import prep_rxn_smi_input


class Datum_agent(NamedTuple):
    """Chemprop dataum with added agent multi-hot vector"""

    a_input: np.ndarray
    mg: MolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


def standardize_and_encode_agents(
    agents: list[AgentRecord], agent_standardizer: AgentStandardizer, agent_encoder: AgentEncoder
) -> np.ndarray:
    """Standardize and encode agents to a tensor of indices and scatter to return a multi-hot tensor"""
    agent_smiles = [agent.smiles for agent in agents]
    standardized_smiles = agent_standardizer.standardize(agent_smiles)
    a_idxs = np.array(agent_encoder.encode(standardized_smiles))
    # output = np.zeros(len(agent_encoder))
    # output[a_idxs] = 1
    return a_idxs


def rxn_smiles_to_mols(rxn_smiles: str) -> tuple[Chem.Mol, Chem.Mol]:
    """prep_rxn_smi_input has already deleted agent and |f| part"""
    rct_smi, agt_smi, pdt_smi = rxn_smiles.split(">")
    rct_smi = f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi  #
    rct = make_mol(rct_smi, keep_h=False, add_h=False)
    pdt = make_mol(pdt_smi, keep_h=False, add_h=False)
    return (rct, pdt)


def indices_to_multihot(indices: np.ndarray, n_agents: int) -> np.ndarray:
    """Convert indices to boolean vector"""
    vector = np.zeros(n_agents, dtype=np.bool_)
    if len(indices) > 0:
        vector[indices] = True
    return vector


class AugmentedAgentReactionDatasetWithReactionClass(Dataset):
    """Lazy augmentation on the fly with index mapping"""

    def __init__(
        self,
        original_data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        rxn_encoder: ReactionClassEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        sample_weighting: str = "pascal",
        **kwargs,
    ):
        self.sample_weighting = sample_weighting
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.rxn_encoder = rxn_encoder
        self.featurizer = featurizer
        self.original_data = original_data
        self.index_mapping = self._create_index_mapping()

    def _create_index_mapping(self):
        """Create mapping from augmented index to (original_idx, (r, comb_idx))
        For base case, (original_idx, -1)
        """
        mapping = []
        for orig_idx, datum in enumerate(self.original_data):
            a_idxs_orig = standardize_and_encode_agents(
                datum.agents, self.agent_standardizer, self.agent_encoder
            )

            # base case
            mapping.append((orig_idx, -1))

            # combinations
            for r in range(1, len(a_idxs_orig) + 1):
                for combo_idx, _ in enumerate(combinations(range(len(a_idxs_orig)), r)):
                    mapping.append((orig_idx, (r, combo_idx)))
        return mapping

    def pascal_triangle_weights(self, n_agents: int) -> np.ndarray:
        """Generates weights based on Pascal's Triangle for a given number of agents."""
        weights = [math.comb(n_agents, k) for k in range(n_agents + 1)]
        inv_weights = [1 / w for w in weights]
        return np.array(inv_weights)

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> Datum_agent:
        orig_idx, aug_spec = self.index_mapping[idx]
        original_datum = self.original_data[orig_idx]

        # get all required indices and weights
        a_idxs_orig = standardize_and_encode_agents(
            original_datum.agents, self.agent_standardizer, self.agent_encoder
        )
        if self.sample_weighting == "pascal":
            all_sample_weights = self.pascal_triangle_weights(len(a_idxs_orig))
        elif self.sample_weighting == "uniform":
            all_sample_weights = np.full((2 ** len(a_idxs_orig),), 1 / (2 ** len(a_idxs_orig)))
        elif self.sample_weighting == "none":
            all_sample_weights = np.full((2 ** len(a_idxs_orig),), 1)
        else:
            raise ValueError(f"Invalid sample_weighting: {self.sample_weighting}")

        if aug_spec == -1:
            input_indices = a_idxs_orig.copy()
            target_indices = np.array([0])
            weight = 1.0
        else:
            r, combo_idx = aug_spec
            all_combos = list(combinations(range(len(a_idxs_orig)), r))
            combo = all_combos[combo_idx]
            input_indices = np.delete(a_idxs_orig, combo)
            target_indices = a_idxs_orig[list(combo)]
            weight = all_sample_weights[r]

        rxn_smiles = prep_rxn_smi_input(original_datum.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)

        if rxn_mols[0] is None or rxn_mols[1] is None:
            # logger.warning(f"Found invalid reaction, using placeholder: {original_datum.rxn_smiles}")
            dummy_mg = self.featurizer((Chem.MolFromSmiles("CCCCC"), Chem.MolFromSmiles("CCCCC")))
            return Datum_agent(
                a_input=indices_to_multihot(input_indices, len(self.agent_encoder)),
                mg=dummy_mg,  # dummy mg
                V_d=None,
                x_d=None,
                y=np.zeros(len(self.agent_encoder), dtype=np.bool_),  # dummy y
                weight=0.0,
                lt_mask=None,
                gt_mask=None,
            )

        mg = self.featurizer(rxn_mols)
        return Datum_agent(
            a_input=indices_to_multihot(input_indices, len(self.agent_encoder)),
            mg=mg,
            V_d=None,
            x_d=self.rxn_encoder.to_onehot(original_datum.rxn_class),
            y=indices_to_multihot(target_indices, len(self.agent_encoder)),
            weight=weight,
            lt_mask=None,
            gt_mask=None,
        )


class AgentReactionDatasetWithReactionClass(Dataset):
    """Agent reaction dataset with no augmentation, for evaluation"""

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        rxn_encoder: ReactionClassEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        **kwargs,
    ):
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.rxn_encoder = rxn_encoder
        self.featurizer = featurizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Datum_agent:
        d = self.data[idx]

        # base case: input none, target full set
        a_idxs = standardize_and_encode_agents(
            d.agents, self.agent_standardizer, self.agent_encoder
        )

        # mg
        rxn_smiles = prep_rxn_smi_input(d.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)

        if rxn_mols[0] is None or rxn_mols[1] is None:
            # logger.warning(f"Found invalid reaction, using placeholder: {d.rxn_smiles}")
            dummy_mg = self.featurizer((Chem.MolFromSmiles("CCCCC"), Chem.MolFromSmiles("CCCCC")))
            return Datum_agent(
                a_input=np.zeros(len(self.agent_encoder), dtype=np.bool_),
                mg=dummy_mg,  # dummy mg
                V_d=None,
                x_d=None,
                y=np.zeros(len(self.agent_encoder), dtype=np.bool_),  # dummy y
                weight=0.0,
                lt_mask=None,
                gt_mask=None,
            )

        mg = self.featurizer(rxn_mols)
        return Datum_agent(
            a_input=indices_to_multihot([], len(self.agent_encoder)),
            mg=mg,
            V_d=None,
            x_d=self.rxn_encoder.to_onehot(d.rxn_class),
            y=indices_to_multihot(a_idxs, len(self.agent_encoder)),
            weight=1.0,
            lt_mask=None,
            gt_mask=None,
        )


class GNNBinnedTemperatureDataset(Dataset):
    """Dataset for temperature prediction, formulated as a binned classification task"""

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        bins: None | list[float] = None,
        **kwargs,
    ):
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.featurizer = featurizer

        if bins is None:
            bins = np.arange(-100, 201, 10) + 273.15
        self.bins = bins

        self.data = data
        super().__init__()

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

    def _indices_to_multihot(self, indices: np.ndarray) -> np.ndarray:
        """Convert indices to boolean vector"""
        vector = np.zeros(len(self.agent_encoder), dtype=np.bool_)
        if len(indices) > 0:
            vector[indices] = True
        return vector

    def __getitem__(self, idx) -> Datum_agent:
        d = self.data[idx]  # ReactionDatum

        a_input = self._indices_to_multihot(
            standardize_and_encode_agents(d.agents, self.agent_standardizer, self.agent_encoder)
        )
        y = np.digitize(d.temperature, self.bins)
        rxn_smiles = prep_rxn_smi_input(d.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)
        mg = self.featurizer(rxn_mols)

        return Datum_agent(
            a_input=a_input,
            mg=mg,
            V_d=None,
            x_d=None,
            y=y,
            weight=1.0,
            lt_mask=None,
            gt_mask=None,
        )

    def __len__(self):
        return len(self.data)


MAX_NUM_REACTANTS = 5


class GNNBinnedReactantAmountDataset(Dataset):
    """Processes reaction data to predict binned reactant amounts,
    generating molecular fingerprints and graphs on-the-fly.

    Args:
    data  : list[ReactionDatum]
        List of reaction records containing SMILES and amounts
    agent_standardizer : AgentStandardizer
        standardizer for agent SMILES
    agent_encoder : AgentEncoder
        encoder for converting agents to indices
    featurizer : CondensedGraphOfReactionFeaturizer
        graph featurizer for reactions
    morgan_generator : FingerprintGenerator64
        morgan fingerprint generator
    bins : list[float], optional, default=[0.95, 1.05, ..., 7.5]
        bin edges for discretizing amounts.

    Returns:
        Datum_agent: A named tuple containing:
            - a_input : torch.Tensor  (batch_size, num_agents)
                agent multi-hot encoding [n_possible_agents], denoted by len(agent_encoder).
            - mg : MolecularGraph
                graph representation of the reaction
            - V_d : None
                placeholder for future features
            - x_d : np.ndarray (batch_size, MAX_NUM_REACTANTS, 2048)
                reactant fingerprints [MAX_NUM_REACTANTS, 2048]
            - y : np.ndarray (batch_size, MAX_NUM_REACTANTS)
                binned relative amounts [MAX_NUM_REACTANTS]
            - weight : float
                sample weight, currently 1.0
            - lt_mask : None
                placeholder for future features
            - gt_mask (None): Placeholder for future features

    """

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        # morgan_generator: rdFingerprintGenerator,
        bins: None | list[float] = None,
        fp_radius: int = 3,
        fp_length: int = 2048,
        **kwargs,
    ):
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.featurizer = featurizer
        # self.morgan_generator = morgan_generator

        if bins is None:
            bins = [0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5]
        self.bins = bins
        self.fp_radius = fp_radius
        self.fp_length = fp_length

        self.data = data
        super().__init__()

    def generate_fingerprint(self, agent_records: list[AgentRecord]) -> np.ndarray:
        # features_combined = np.zeros((2048,), dtype=bool)
        # for agent_record in agent_records:
        #     mol = Chem.MolFromSmiles(agent_record.smiles)
        #     if mol is not None and mol.GetNumHeavyAtoms() > 0:
        #         fp = self.morgan_generator.GetFingerprint(mol)
        #         arr = np.zeros((2048,), dtype=bool)
        #         DataStructs.ConvertToNumpyArray(fp, arr)
        #         features_combined |= arr
        # return features_combined
        smi_list = [agent_record.smiles for agent_record in agent_records]
        merged_smi = ".".join(smi_list)
        mol = Chem.MolFromSmiles(merged_smi)

        fp_arr = np.zeros((2048,), dtype=bool)
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            # fp_arr = self.morgan_generator.GetFingerprint(mol)
            fp_arr = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.fp_radius, nBits=self.fp_length
            )
        return torch.tensor(fp_arr, dtype=torch.bool)

    def __getitem__(self, idx) -> Datum_agent:
        d = self.data[idx]

        # Process agents
        a_input = indices_to_multihot(
            indices=standardize_and_encode_agents(
                d.agents, self.agent_standardizer, self.agent_encoder
            ),
            n_agents=len(self.agent_encoder),
        )

        # Process reactants
        FP_reactants = np.stack(
            [self.generate_fingerprint([reactant]) for reactant in d.reactants[:MAX_NUM_REACTANTS]]
        )

        # Calculate relative amounts
        limiting_reactant_amount = min([reactant.amount for reactant in d.reactants])
        r_amount = [
            reactant.amount / limiting_reactant_amount
            for reactant in d.reactants[:MAX_NUM_REACTANTS]
        ]
        r_amount_target = np.digitize(r_amount, self.bins)

        # Pad arrays if needed
        if FP_reactants.shape[0] < MAX_NUM_REACTANTS:
            FP_reactants = np.pad(
                FP_reactants,
                ((0, MAX_NUM_REACTANTS - len(FP_reactants)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            r_amount_target = np.pad(
                r_amount_target,
                (0, MAX_NUM_REACTANTS - len(r_amount_target)),
                mode="constant",
                constant_values=0,
            )

        # Generate molecular graph
        rxn_smiles = prep_rxn_smi_input(d.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)

        if rxn_mols[0] is None or rxn_mols[1] is None:
            logger.warning(f"Found invalid reaction, using placeholder: {d.rxn_smiles}")

            mg = self.featurizer((Chem.MolFromSmiles("CCCCC"), Chem.MolFromSmiles("CCCCC")))
            return Datum_agent(
                a_input=a_input,
                mg=mg,  # dummy mg
                V_d=None,
                x_d=np.zeros((MAX_NUM_REACTANTS, 2048), dtype=bool),  # dummy FP, match other fp
                y=np.zeros(MAX_NUM_REACTANTS, dtype=np.int64),  # dummy y
                weight=0.0,  # Set weight to 0 to exclude from loss calculation
                lt_mask=None,
                gt_mask=None,
            )

        mg = self.featurizer(rxn_mols)
        return Datum_agent(
            a_input=a_input,
            mg=mg,
            V_d=None,
            x_d=FP_reactants,
            y=r_amount_target,
            weight=1.0,
            lt_mask=None,
            gt_mask=None,
        )

    def __len__(self):
        return len(self.data)


class GNNBinnedAgentAmountOneShotDataset(Dataset):
    """
    Dataset for agent amount prediction with lazy evaluation

    Args:
    data : list[ReactionDatum]
        list of reaction records containing SMILES and amounts
    agent_standardizer : AgentStandardizer
        standardizer for agent SMILES
    agent_encoder : AgentEncoder
        encoder for converting agents to indices
    featurizer : CondensedGraphOfReactionFeaturizer
        graph featurizer for reactions
    bins : list[float], optional, default using small, regular, large bins
        bin edges for discretizing amounts.

    Returns:
        Datum_agent: A named tuple containing:
            - a_input : torch.Tensor  (batch_size, num_agents)
                agent multi-hot encoding [n_possible_agents], denoted by len(agent_encoder).
            - mg : MolecularGraph
                graph representation of the reaction
            - V_d : None
                placeholder for future features
            - x_d : None
            - y : np.ndarray (batch_size, num_bins)
                binned relative amounts,
            - weight : float
                sample weight, currently 1.0
    """

    def __init__(
        self,
        data: list[ReactionDatum],
        agent_standardizer: AgentStandardizer,
        agent_encoder: AgentEncoder,
        featurizer: CondensedGraphOfReactionFeaturizer,
        bins: None | list[float] = None,
        **kwargs,
    ):
        self.agent_standardizer = agent_standardizer
        self.agent_encoder = agent_encoder
        self.featurizer = featurizer

        if bins is None:
            small_bins = np.array([0, 0.075, 0.15, 0.25, 0.55, 0.95])
            regular_bins = np.array([1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            large_bins = np.array([15.5, 25.5, 35.5, 45.5, 55.5, 65.5, 75.5, 85.5, 100.5])
            bins = np.concatenate([small_bins, regular_bins, large_bins])
        self.bins = bins

        self.data = data
        super().__init__()

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

    def __getitem__(self, idx) -> Datum_agent:
        d = self.data[idx]

        # Process agents
        a_input = indices_to_multihot(
            standardize_and_encode_agents(d.agents, self.agent_standardizer, self.agent_encoder),
            len(self.agent_encoder),
        )  # (batch_size, num_agents) multi-hot

        # Calculate relative amounts and bin assignments
        limiting_reactant_amount = min([reactant.amount for reactant in d.reactants])
        a_amount_target = np.zeros(
            len(a_input)
        )  # (batch_size, num_agents) -> correct bin idxs on existing agents, for non-existing agents, it's target bin is 0

        for i, agent in enumerate(d.agents):
            standardized_smiles = self.agent_standardizer.standardize([agent.smiles])
            agent_idx = self.agent_encoder.encode(standardized_smiles)[0]
            relative_amount = agent.amount / limiting_reactant_amount
            a_amount_target[agent_idx] = np.digitize([relative_amount], self.bins)[0]

        # Generate molecular graph
        rxn_smiles = prep_rxn_smi_input(d.rxn_smiles)
        rxn_mols = rxn_smiles_to_mols(rxn_smiles)  # (rct, pdt)
        mg = self.featurizer(rxn_mols)

        return Datum_agent(
            a_input=a_input,
            mg=mg,
            V_d=None,
            x_d=None,
            y=a_amount_target,
            weight=1.0,
            lt_mask=None,
            gt_mask=None,
        )

    def __len__(self):
        return len(self.data)
