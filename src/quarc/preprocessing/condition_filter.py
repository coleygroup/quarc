"""
Stage-specific condition filters

Stage 1 (Agent Data)
* Agent existence
* [included for new stage 1] Agent amount existence [excluds 47% data]
* [included for new stage 1] Unique agent [as a QC check]

Stage 2 (Temperature Data):
* Temperature existence
* Ranger filter (-100°C to 200°C)

Stage 3 (Reactant Amount Data):
* [not included] Multiple reactants (>1) [removed bc this excluds 25% data, and single reactant is fine]
* Reactant amount existence
* Unique reactant
* Reactant ratio range (0.001 to 7)

Stage 4 (Agent Amount Data):
* Agent existence check
* Agent amount existence
* Reactant amount existence
* Unique agent
* Agent ratio range (0.001 to 1000)
* Unwanted non-solvent: large ratios but not solvents

"""

import json
import pickle
from pathlib import Path
from typing import Callable

from loguru import logger
from rdkit import RDLogger
from tqdm import tqdm

from quarc.data.datapoints import ReactionDatum
from quarc.utils.smiles_utils import get_common_solvents_canonical

COMMON_SOLVENTS_CANONICAL = get_common_solvents_canonical()

RDLogger.DisableLog("rdApp.*")
# Filters: True -> keep, False -> skip


## Stage 1: Agent Data
def filter_by_agent_existence(rxn: ReactionDatum) -> bool:
    """skip if no agents"""
    return len(rxn.agents) > 0


def filter_by_unique_agent(rxn: ReactionDatum) -> bool:
    """skip if not all agents are unique"""
    return len(set([a.smiles for a in rxn.agents])) == len(rxn.agents)


def filter_by_agent_amount_existence(rxn: ReactionDatum) -> bool:
    """skip if any AgentRecord has no amount (strict!!)"""
    return all(agent.amount is not None for agent in rxn.agents)


## stage 2: temp filter
def filter_by_temperature(rxn: ReactionDatum) -> bool:
    """skip if temperature is None"""
    return rxn.temperature is not None


def filter_by_temperature_range(
    rxn: ReactionDatum, lower_bound: float, upper_bound: float
) -> bool:
    """skip if temperature is not in range -100 to 200"""
    return lower_bound <= (rxn.temperature - 273.15) <= upper_bound


## stage 3: reactant amount filter
def filter_by_reactant_num(rxn: ReactionDatum) -> bool:
    """skip if num reactant <= 1 (=0 should be hanlded in initial filter). No need to predict amount if only 1 reactant"""
    return len(rxn.reactants) > 1


def filter_by_reactant_amount_existence(
    rxn: ReactionDatum,
) -> bool:  # add this to agent_amount fitler as well for trustful limiting reactant
    """skip if any reactant AgentRecord has no amount (this is very strict!)"""
    return all(reactant.amount is not None for reactant in rxn.reactants)


def filter_by_reactant_ratio_range(
    rxn: ReactionDatum, min_ratio: float = 0.001, max_ratio: float = 7
) -> bool:
    """skip if max ratio > 7"""
    r_amounts = [r.amount for r in rxn.reactants]
    limiting_amount = min(r_amounts)
    reactant_ratios = [amount / limiting_amount for amount in r_amounts]
    return all(min_ratio <= ratio <= max_ratio for ratio in reactant_ratios)


def filter_by_unique_reactant(rxn: ReactionDatum) -> bool:
    """skip if not all reactants are unique"""
    return len(set([r.smiles for r in rxn.reactants])) == len(rxn.reactants)


## stage 4: agent amount filters
# reuse agent existence, agent amount, reactant amount


def filter_by_agent_ratio_range(
    rxn: ReactionDatum, min_ratio: float = 0.001, max_ratio: float = 1000
) -> bool:
    """skip if any agent amount is 0"""
    limiting_reactant_amount = min(r.amount for r in rxn.reactants)
    a_ratios = [a.amount / limiting_reactant_amount for a in rxn.agents]
    return all([min_ratio <= ratio <= max_ratio for ratio in a_ratios])


def filter_by_nonsolvent_high_ratio(rxn: ReactionDatum) -> bool:
    """skip if any non-solvent with ratio > 10 (only solvents are allowed to exceed 10)"""
    limiting_reactant_amount = min(r.amount for r in rxn.reactants)

    for agent in rxn.agents:
        ratio = agent.amount / limiting_reactant_amount
        is_solvent = agent.smiles in COMMON_SOLVENTS_CANONICAL
        if ratio > 10 and not is_solvent:
            return False
    return True


def apply_filter_group(
    data: list[ReactionDatum], filters: list[tuple[Callable, str]], group_name: str
) -> list[ReactionDatum]:
    """Apply filter and log.
    Filter function returns: True -> keep, False -> remove
    """
    initial_count = len(data)

    filtered_data = []
    for datum in tqdm(data, desc=f"{group_name} filtering"):
        for filter_func, _filter_name in filters:
            if not filter_func(datum):
                break
        else:
            filtered_data.append(datum)

    final_count = len(filtered_data)
    stats = {
        "len_before": initial_count,
        "len_after": final_count,
        "removed_ratio": round((initial_count - final_count) / initial_count, 3),
    }
    data.clear()

    logger.info(f"{group_name} filtering results:\n" + json.dumps(stats, indent=2))
    return filtered_data, stats


def run_stage1_filters(config):
    """Run stage 1 (Agent Data) filters"""
    logger.info("---Running stage 1 filters---")
    # Load paths from config
    input_dir = Path(config["stage_1_filter"]["input_dir"])
    output_dir = Path(config["stage_1_filter"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    filters = [
        (filter_by_agent_existence, "agent_existence"),  # removing ~0.4%
        (filter_by_unique_agent, "unique_agent"),
        (filter_by_agent_amount_existence, "agent_amount"),  # removing ~47%
    ]
    filter_metadata = {
        "filters_applied": [name for _, name in filters],
        "stats": {"train": {}, "val": {}, "test": {}},
    }

    # Load data
    for split in ["train", "val", "test"]:
        input_path = input_dir / f"full_{split}.pickle"
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        # Apply filters
        filtered_data, stats = apply_filter_group(data, filters, f"stage_1_{split}")
        filter_metadata["stats"][split] = stats

        # Save filtered data
        output_path = output_dir / f"stage1_{split}.pickle"
        with open(output_path, "wb") as f:
            pickle.dump(filtered_data, f, pickle.HIGHEST_PROTOCOL)

        del data, filtered_data

    # save filter meta data
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(filter_metadata, f, indent=2)


def run_stage2_filters(config):
    """Run stage 2 (Temperature Data) filters"""
    logger.info("---Running stage 2 filters---")
    input_dir = Path(config["stage_2_filter"]["input_dir"])
    output_dir = Path(config["stage_2_filter"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # unit: Celsius
    temp_range = config["stage_2_filter"]["temperature_range"]

    filters = [
        (filter_by_temperature, "temperature_existence"),
        (
            lambda rxn: filter_by_temperature_range(
                rxn, lower_bound=temp_range["lower"], upper_bound=temp_range["upper"]
            ),
            "temperature_range",
        ),
    ]
    filter_metadata = {
        "filters_applied": [name for _, name in filters],
        "stats": {"train": {}, "val": {}, "test": {}},
    }

    for split in ["train", "val", "test"]:
        input_path = input_dir / f"full_{split}.pickle"
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        filtered_data, stats = apply_filter_group(data, filters, f"stage_2_{split}")
        filter_metadata["stats"][split] = stats

        output_path = output_dir / f"stage2_{split}.pickle"
        with open(output_path, "wb") as f:
            pickle.dump(filtered_data, f, pickle.HIGHEST_PROTOCOL)

        del data, filtered_data

    # save filter meta data
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(filter_metadata, f, indent=2)


def run_stage3_filters(config):
    """Run stage 3 (Reactant Amount Data) filters"""
    logger.info("---Running stage 3 filters---")
    input_dir = Path(config["stage_3_filter"]["input_dir"])
    output_dir = Path(config["stage_3_filter"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    ratio_range = config["stage_3_filter"]["ratio_range"]

    filters = [
        # (filter_by_reactant_num, "multiple_reactants"), throws away ~25% of data (single reactant reactions got thrown away, not good)
        (filter_by_unique_reactant, "unique_reactants"),
        (filter_by_reactant_amount_existence, "reactant_amount_existence"),
        (
            lambda rxn: filter_by_reactant_ratio_range(
                rxn, min_ratio=ratio_range["lower"], max_ratio=ratio_range["upper"]
            ),
            "reactant_ratio_range",
        ),
    ]
    filter_metadata = {
        "filters_applied": [name for _, name in filters],
        "stats": {"train": {}, "val": {}, "test": {}},
    }

    for split in ["train", "val", "test"]:
        input_path = input_dir / f"full_{split}.pickle"
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        filtered_data, stats = apply_filter_group(data, filters, f"stage_3_{split}")
        filter_metadata["stats"][split] = stats

        output_path = output_dir / f"stage3_{split}.pickle"
        with open(output_path, "wb") as f:
            pickle.dump(filtered_data, f, pickle.HIGHEST_PROTOCOL)

        del data, filtered_data

    # save filter meta data
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(filter_metadata, f, indent=2)


def run_stage4_filters(config):
    """Run stage 4 (Reactant Amount Data) filters"""
    logger.info("---Running stage 4 filters---")
    input_dir = Path(config["stage_4_filter"]["input_dir"])
    output_dir = Path(config["stage_4_filter"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    ratio_range = config["stage_4_filter"]["ratio_range"]

    filters = [
        (filter_by_agent_existence, "agent_existence"),
        (filter_by_agent_amount_existence, "agent_amount_existence"),
        (filter_by_reactant_amount_existence, "reactant_amount_existence"),
        (filter_by_unique_agent, "unique_agents"),
        (
            lambda rxn: filter_by_agent_ratio_range(
                rxn, min_ratio=ratio_range["lower"], max_ratio=ratio_range["upper"]
            ),
            "agent_ratio_range",
        ),
        (filter_by_nonsolvent_high_ratio, "nonsolvent_high_ratio"),
    ]

    filter_metadata = {
        "filters_applied": [name for _, name in filters],
        "stats": {"train": {}, "val": {}, "test": {}},
    }

    for split in ["train", "val", "test"]:
        input_path = input_dir / f"full_{split}.pickle"
        with open(input_path, "rb") as f:
            data = pickle.load(f)

        filtered_data, stats = apply_filter_group(data, filters, f"stage_4_{split}")
        filter_metadata["stats"][split] = stats

        output_path = output_dir / f"stage4_{split}.pickle"
        with open(output_path, "wb") as f:
            pickle.dump(filtered_data, f, pickle.HIGHEST_PROTOCOL)

        del data, filtered_data

    # save filter meta data
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(filter_metadata, f, indent=2)
