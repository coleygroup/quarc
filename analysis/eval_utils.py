from dataclasses import dataclass, field
from typing import List, Any, Tuple, Literal, Dict, Callable
from tqdm import tqdm
import pandas as pd


from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.analysis.correctness import get_criteria_fn

@dataclass
class BasePrediction:
    """
    Base class for a self-contained prediction result.
    Contains the target and metadata common to all stages.
    """

    doc_id: str
    rxn_class: str
    rxn_smiles: str


@dataclass
class Stage1Prediction(BasePrediction):
    """Contains predictions for Stage 1 (Agents)."""

    predictions: list[list[int]]  # list of top-kpredicted agent sets from beam search
    target_indices: list[int]  # Target agent indices


@dataclass
class Stage2Prediction(BasePrediction):
    """Contains predictions for Stage 2 (Temperature)."""

    prediction: int  # A single predicted temperature bin
    target: int


@dataclass
class Stage3Prediction(BasePrediction):
    """Contains predictions for Stage 3 (Reactant Amounts)."""

    predictions: list[tuple[str, int]]  # list of (reactant_smiles, predicted_bin)
    target: list[tuple[str, int]]  # list of (reactant_smiles, true_bin)


@dataclass
class Stage4Prediction(BasePrediction):
    """Contains predictions for Stage 4 (Agent Amounts)."""

    predictions: list[tuple[int, int]]  # list of (agent_index, predicted_bin)
    target: list[tuple[int, int]]  # list of (agent_index, true_bin)


def map_local_to_global_indices(df, class_to_global_map):
    """
    Convert local class-specific indices to global training set indices
    """
    global_indices = []
    for local_indices, class_label in zip(df["neighbors"], df["reaction_class"]):
        global_map = class_to_global_map.get(class_label)
        if not global_map:
            global_indices.append(None)
            continue
        row_global_indices = [global_map[local_idx] for local_idx in local_indices]
        global_indices.append(row_global_indices)
    df["global_neighbors"] = global_indices
    return df


def calculate_nn_matched_distances(
    test_data,
    train_data,
    df_global,
    a_enc,
    a_standardizer,
    criteria: Literal["set", "idx", "combination"] = "set",
) -> List[float]:

    matched_distances = []
    criteria_fn = get_criteria_fn(criteria)

    for i, data in enumerate(tqdm(test_data)):
        assert data.rxn_class == df_global["reaction_class"].iloc[i]

        target_smi_lst = a_standardizer.standardize([agent.smiles for agent in data.agents])
        neighbors = df_global["global_neighbors"].iloc[i]

        if neighbors is None:
            matched_distances.append(-1)
            continue

        match_found = False
        for j, train_idx in enumerate(neighbors):
            train_data_point = train_data[train_idx]
            pred_smi_list = a_standardizer.standardize(
                [agent.smiles for agent in train_data_point.agents]
            )

            if criteria_fn(pred_smi_list, target_smi_lst, a_enc):
                matched_distances.append(df_global["distances"].iloc[i][j])
                match_found = True
                break

        if not match_found:
            matched_distances.append(-1)

    return matched_distances


def calculate_popularity_accuracies(
    test_data,
    top_10_sets,
    a_enc,
    a_standardizer,
    criteria: Literal["set", "idx", "combination"] = "set",
) -> Tuple[Dict[int, float], List[int]]:
    """
    Calculate popularity baseline accuracies.
    Returns tuple: (accuracies, first_match_positions)
    """
    k_hits = {k: 0 for k in range(1, 11)}
    total = len(test_data)
    first_match_positions = []

    criteria_fn = get_criteria_fn(criteria)

    for data in tqdm(test_data, desc=f"Calculating Pop accuracies ({criteria})"):
        target_smi_lst = a_standardizer.standardize([agent.smiles for agent in data.agents])
        top_sets = top_10_sets.get(data.rxn_class, [])

        match_found = False
        if not top_sets:
            first_match_positions.append(-1)
            continue

        for k in range(1, 11):
            if k > len(top_sets):
                break

            pred_smi_list_k = a_enc.decode(top_sets[k - 1][0])

            if criteria_fn(pred_smi_list_k, target_smi_lst, a_enc):
                first_match_positions.append(k - 1)
                match_found = True

                for hit_k in range(k, 11):
                    k_hits[hit_k] += 1
                break

        if not match_found:
            first_match_positions.append(-1)

    accuracies = {k: n_hits / total for k, n_hits in k_hits.items()}
    return accuracies, first_match_positions
