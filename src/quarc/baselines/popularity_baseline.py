from collections import defaultdict, Counter
import numpy as np
import pickle
import sys
from tqdm import tqdm
from typing import NamedTuple, Tuple, FrozenSet

from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.data.binning import BinningConfig
from quarc.settings import load as load_settings

cfg = load_settings()

a_enc = AgentEncoder(class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json")
a_standardizer = AgentStandardizer(
    conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
    other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
)


def get_popularity_sets_from_train(train_data, a_standardizer, a_enc):
    popularity_sets = defaultdict(Counter)
    for data in tqdm(train_data):
        standardized_smiles = a_standardizer.standardize([agent.smiles for agent in data.agents])
        encoded_agent_set = frozenset(a_enc(standardized_smiles))
        popularity_sets[data.rxn_class][encoded_agent_set] += 1

    top_10_sets = {
        rxn_class: counter.most_common(10) for rxn_class, counter in popularity_sets.items()
    }

    with open(cfg.processed_data_dir / "pop_baseline/stage1_top_10_popularity_sets.pickle", "wb") as f:
        pickle.dump(top_10_sets, f)
    return popularity_sets


class CanonicalCondition(NamedTuple):
    binned_reactant_ratios: Tuple[int, ...]
    binned_agents: FrozenSet[Tuple[int, str]]
    binned_temperature: str


def get_canonical_condition(
    datum, agent_standardizer: AgentStandardizer, agent_encoder: AgentEncoder
) -> CanonicalCondition:
    """
    Converts a reaction datum into a canonical condition representation
    that can be counted for popularity analysis.
    """
    binning_config = BinningConfig.default()
    temperature_bins = binning_config.temperature_bins
    reactant_amount_bins = binning_config.reactant_amount_bins
    agent_amount_bins = binning_config.agent_amount_bins

    # Skip reactions with missing data
    if (
        any(r.amount is None for r in datum.reactants)
        or any(a.amount is None for a in datum.agents)
        or datum.temperature is None
    ):
        return None

    min_reactant_amount = min(r.amount for r in datum.reactants)

    binned_reactant_ratios = [
        np.digitize(r.amount / min_reactant_amount, reactant_amount_bins) for r in datum.reactants
    ]
    canonical_binned_reactant_ratios = tuple(sorted(binned_reactant_ratios))

    binned_agents = []
    for agent in datum.agents:
        if agent.smiles is not None:
            standardized_smiles = agent_standardizer.standardize([agent.smiles])
            agent_idx = agent_encoder.encode(standardized_smiles)[0]
            agent_amount = np.digitize(agent.amount / min_reactant_amount, agent_amount_bins)
            binned_agents.append((agent_idx, agent_amount))
    binned_agents.sort(key=lambda x: x[0])
    canonical_binned_agents = frozenset(binned_agents)

    binned_temp = np.digitize(datum.temperature, temperature_bins)

    return CanonicalCondition(
        binned_reactant_ratios=canonical_binned_reactant_ratios,
        binned_agents=canonical_binned_agents,
        binned_temperature=binned_temp,
    )


def get_overall_popularity_from_train(train_data, a_standardizer, a_enc):
    popularity_baseline = defaultdict(Counter)

    for data in tqdm(train_data):
        if not data.rxn_class:
            continue

        condition = get_canonical_condition(data, a_standardizer, a_enc)
        if condition is not None:
            popularity_baseline[data.rxn_class][condition] += 1

    top_10_conditions = {
        rxn_class: counter.most_common(10) for rxn_class, counter in popularity_baseline.items()
    }

    with open(cfg.processed_data_dir / "pop_baseline/top_10_overall_conditions.pickle", "wb") as f:
        pickle.dump(top_10_conditions, f)

    return popularity_baseline