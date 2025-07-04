from dataclasses import dataclass
from collections import Counter
import numpy as np
from typing import Any
import torch
from torch import Tensor
from typing import Callable, Literal

from quarc.predictors.base import StagePrediction
from quarc.models.modules.agent_encoder import AgentEncoder

@dataclass
class StageResult:
    """Results for individual stage"""

    is_correct: bool
    is_correct_relaxed: bool = False  # within-one-bin tolerance
    is_correct_2bins: bool = False  # within-two-bins tolerance


@dataclass
class OverallResult:
    """Overall results for a single reaction"""

    # Overall results
    is_fully_correct: bool
    is_fully_correct_relaxed: bool  # within-one-bin tolerance

    # Per-stage results
    agent_result: StageResult
    temperature_result: StageResult
    reactant_amount_result: StageResult
    agent_amount_result: StageResult

def is_correct_by_set(
    preds: Tensor | list[str] | list[int],
    targets: Tensor | list[str] | list[int],
    a_enc: AgentEncoder,
) -> bool:
    """
    1. use set match instead direct match for each predictions (not entire prediction as a whole set)
    2. if water is the only difference either way, count as correct
        e.g set(one_pred) - set(target) = 'O' or set(target) - set(one_pred) = 'O'

    Params:
        preds: 1d tensor (multi-hot encoded) or list of smiles (str) or list of indices (int)
        targets:  1d tensor (multi-hot encoded) or list of smiles (str) or list of indices (int)

    """

    if not preds or not targets:
        return False

    def _decode_input_to_smiles(input_data, a_enc):
        if isinstance(input_data, list) and isinstance(input_data[0], str):  # list of SMILES
            return input_data
        elif isinstance(input_data, Tensor):  # multi-hot vector
            input_data = input_data.squeeze() if input_data.shape[0] == 1 else input_data
            return a_enc.decode(torch.nonzero(input_data).squeeze(-1).tolist())
        elif isinstance(input_data, list) and isinstance(input_data[0], int):  # list of indices
            return a_enc.decode(input_data)
        else:
            raise TypeError(
                "Input must be either a tensor, list of indices, or list of SMILES strings"
            )

    pred_smi = _decode_input_to_smiles(preds, a_enc)
    target_smi = _decode_input_to_smiles(targets, a_enc)

    # generate set of smiles, make set by splitting by '.'
    pred_smi_set = set(
        [constituent for agent in [a.split(".") for a in pred_smi] for constituent in agent]
    )
    target_smi_set = set(
        [constituent for agent in [a.split(".") for a in target_smi] for constituent in agent]
    )

    # check if set match
    return (pred_smi_set == target_smi_set) or (pred_smi_set ^ target_smi_set == {"O"})


def is_correct_by_idx(
    preds: Tensor | list[str] | list[int],
    targets: Tensor | list[str] | list[int],
    a_enc: AgentEncoder = None,
) -> bool:
    """
    Check if two tensors are equal by comparing indices of non-zero elements.
    """

    if not preds or not targets:
        return False

    def _decode_input_to_indices(input_data, a_enc) -> list[int]:
        if isinstance(input_data, list) and isinstance(input_data[0], int):
            return input_data
        elif isinstance(input_data, list) and isinstance(input_data[0], str):
            # input_data = input_data.squeeze() if input_data.shape[0] == 1 else input_data
            return a_enc.encode(input_data)
        elif isinstance(input_data, Tensor):
            return torch.nonzero(input_data).squeeze(-1).tolist()
        else:
            raise TypeError(
                "Input must be either a tensor, list of indices, or list of SMILES strings"
            )

    if isinstance(preds, Tensor) and isinstance(targets, Tensor):
        return torch.equal(preds, targets)

    pred_idx = _decode_input_to_indices(preds, a_enc)
    target_idx = _decode_input_to_indices(targets, a_enc)

    return set(pred_idx) == set(target_idx)


def get_criteria_fn(
    criteria: Literal["set", "idx"],
) -> Callable[[Tensor, Tensor, AgentEncoder], bool]:
    """Get the criteria function (correct by set/idx). Each criteria_fn will take preds, targets, and a_enc.
    Note: a_enc will only be used for "set", but it is passed to all criteria_fn for consistency.
    """
    if criteria == "set":
        return is_correct_by_set
    elif criteria == "idx":
        return is_correct_by_idx
    else:
        raise TypeError("criteria must be either 'set', or 'idx'")


def check_agent_identity(predicted: list[int], target: list[int], agent_encoder) -> StageResult:
    """Evaluate agent identity prediction. is_correct is the strict check, is_correct_relaxed is the relaxed check."""
    # requires index match
    is_correct = is_correct_by_idx(predicted, target, agent_encoder)

    # set match for SMILES broken down by "."
    is_correct_relaxed = is_correct_by_set(predicted, target, agent_encoder)
    return StageResult(is_correct=is_correct, is_correct_relaxed=is_correct_relaxed)


def check_temperature(predicted: int, target: int) -> StageResult:
    """Evaluate temperature prediction using bin indices

    Args:
        predicted: bin_idx,
        target: bin_idx
    """
    is_correct = predicted == target
    is_within_one = abs(predicted - target) <= 1
    is_within_two = abs(predicted - target) <= 2

    return StageResult(
        is_correct=is_correct, is_correct_relaxed=is_within_one, is_correct_2bins=is_within_two
    )


def check_reactant_amounts(predicted: list[int], target: list[int]) -> StageResult:
    """Evaluate reactant amount prediction using reactant amount bin indices (unordered)

    Args:
        predicted: list[bin_idx]
        target: list[bin_idx]
    """
    is_correct = Counter(predicted) == Counter(target)

    # only model predictions guarantee the same length, baseline predictions may not
    if len(predicted) == len(target):
        is_within_one = all(np.abs(np.array(sorted(predicted)) - np.array(sorted(target))) <= 1)
        is_within_two = all(np.abs(np.array(sorted(predicted)) - np.array(sorted(target))) <= 2)
    else:
        is_within_one = False
        is_within_two = False

    return StageResult(
        is_correct=is_correct, is_correct_relaxed=is_within_one, is_correct_2bins=is_within_two
    )

def check_reactant_amounts_strict(predicted: list[int], target: list[int]) -> StageResult:
    """Evaluate reactant amount prediction using reactant amount bin indices (unordered)
    """
    is_correct = predicted == target
    return StageResult(is_correct=is_correct)


def check_agent_amounts(
    predicted: list[tuple[int, int]], target: list[tuple[int, int]]
) -> StageResult:
    """Evaluate agent amount prediction using mapped agent indices and bin indices

    predicted: list[(agent_idx, bin_idx)]
    target: list[(agent_idx, bin_idx)]
    """
    # Sort both by agent index for comparison
    pred_sorted = sorted(predicted, key=lambda x: x[0])
    target_sorted = sorted(target, key=lambda x: x[0])

    is_correct = pred_sorted == target_sorted

    # Only check within-bin if agent sets match AND have same length
    is_within_one = False
    is_within_two = False
    if (set(p[0] for p in predicted) == set(t[0] for t in target) and
        len(predicted) == len(target)):
        pred_bins = [x[1] for x in pred_sorted]
        target_bins = [x[1] for x in target_sorted]
        is_within_one = all(np.abs(np.array(pred_bins) - np.array(target_bins)) <= 1)
        is_within_two = all(np.abs(np.array(pred_bins) - np.array(target_bins)) <= 2)

    return StageResult(
        is_correct=is_correct, is_correct_relaxed=is_within_one, is_correct_2bins=is_within_two
    )


def check_overall_prediction(
    stage_pred: StagePrediction,
    targets: dict[str, Any],
    agent_encoder,
) -> OverallResult:
    """
    Check all stages of a prediction against targets.
    Uses existing targets from ReactionInput.
    """

    # Stage 1: Agents
    agent_result = check_agent_identity(stage_pred.agents, targets["target_agents"], agent_encoder)

    # Stage 2: Temperature
    temp_result = check_temperature(stage_pred.temp_bin, targets["target_temp"])

    # Stage 3: Reactant amounts
    reactant_result = check_reactant_amounts(
        stage_pred.reactant_bins, targets["target_reactant_amounts"]
    )

    # Stage 4: Agent amounts
    agent_amount_result = check_agent_amounts(
        stage_pred.agent_amount_bins, targets["target_agent_amounts"]
    )

    # Standard stage 1 evaluation uses relaxed check (is_correct_by_set)
    # in overall correctness, this will be overridden by stage 4 bc it requires strict agent idx match
    is_fully_correct = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct
        and reactant_result.is_correct
        and agent_amount_result.is_correct
    )

    # relaxed is not the most representative bc counter/set can be weird
    is_fully_correct_relaxed = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct_relaxed
        and reactant_result.is_correct_relaxed
        and agent_amount_result.is_correct_relaxed
    )

    return OverallResult(
        is_fully_correct=is_fully_correct,
        is_fully_correct_relaxed=is_fully_correct_relaxed,
        agent_result=agent_result,
        temperature_result=temp_result,
        reactant_amount_result=reactant_result,
        agent_amount_result=agent_amount_result,
    )


def check_overall_prediction_strict(
    stage_pred: StagePrediction,
    targets: dict[str, Any],
    agent_encoder,
) -> OverallResult:
    """
    Check all stages of a prediction against targets.
    Uses existing targets from ReactionInput.
    """

    # Stage 1: Agents
    agent_result = check_agent_identity(stage_pred.agents, targets["target_agents"], agent_encoder)

    # Stage 2: Temperature
    temp_result = check_temperature(stage_pred.temp_bin, targets["target_temp"])

    # Stage 3: Reactant amounts
    reactant_result = check_reactant_amounts(
        stage_pred.reactant_bins, targets["target_reactant_amounts"]
    )

    # Stage 4: Agent amounts
    agent_amount_result = check_agent_amounts(
        stage_pred.agent_amount_bins, targets["target_agent_amounts"]
    )

    # Standard stage 1 evaluation uses relaxed check (is_correct_by_set)
    # in overall correctness, this will be overridden by stage 4 bc it requires strict agent idx match
    is_fully_correct = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct
        and reactant_result.is_correct
        and agent_amount_result.is_correct
    )

    # relaxed is not the most representative bc counter/set can be weird
    is_fully_correct_relaxed = (
        agent_result.is_correct_relaxed
        and temp_result.is_correct_relaxed
        and reactant_result.is_correct_relaxed
        and agent_amount_result.is_correct_relaxed
    )

    return OverallResult(
        is_fully_correct=is_fully_correct,
        is_fully_correct_relaxed=is_fully_correct_relaxed,
        agent_result=agent_result,
        temperature_result=temp_result,
        reactant_amount_result=reactant_result,
        agent_amount_result=agent_amount_result,
    )
