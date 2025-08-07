import argparse
import pickle
import random
import sys
from pathlib import Path
from loguru import logger
import torch
import optuna
from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.data.eval_datasets import EvaluationDatasetFactory, UnifiedEvaluationDataset
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.base import PredictionList
from quarc.predictors.multistage_predictor import EnumeratedPredictor
from quarc.settings import load as load_settings
from quarc.cli.quarc_parser import add_opt_opts

from analysis.correctness import check_overall_prediction

cfg = load_settings()


def load_validation_data(data_path: str, sample_size: int = 1000) -> list:
    """Load and sample validation data."""
    with open(data_path, "rb") as f:
        val_data = pickle.load(f)

    if sample_size > 0 and sample_size < len(val_data):
        random.seed(42)
        indices = random.sample(range(len(val_data)), sample_size)
        val_data = [val_data[i] for i in sorted(indices)]
        logger.info(f"Sampled {len(val_data)} reactions for optimization")
    else:
        logger.info(f"Using all {len(val_data)} reactions for optimization")

    return val_data


def load_encoders(cfg):
    agent_encoder = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )
    agent_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )
    rxn_encoder = ReactionClassEncoder(class_path=cfg.pistachio_namerxn_path)
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")
    return agent_encoder, agent_standardizer, rxn_encoder, featurizer


def calculate_topk_accuracy(
    prediction_lists: list[PredictionList],
    reaction_inputs: UnifiedEvaluationDataset,
    agent_encoder: AgentEncoder,
    max_k: int = 10,
):

    hit_counters = {
        "overall": {k: 0 for k in range(1, max_k + 1)},
        "overall_relaxed": {k: 0 for k in range(1, max_k + 1)},
    }

    total_reactions = len(prediction_lists)

    for pred_list, reaction_input in zip(prediction_lists, reaction_inputs):
        match_found = {criterion: False for criterion in hit_counters.keys()}

        for i, stage_pred in enumerate(pred_list.predictions[:max_k]):
            k = i + 1  # 1-indexed k

            # Check correctness
            targets = reaction_input.targets
            correctness = check_overall_prediction(stage_pred, targets, agent_encoder)
            criteria_checks = {
                "overall": correctness.is_fully_correct,
                "overall_relaxed": correctness.is_fully_correct_relaxed,
            }

            for criterion, is_correct in criteria_checks.items():
                if is_correct and not match_found[criterion]:
                    match_found[criterion] = True
                    # Update ALL k values from current position onward
                    for hit_k in range(k, max_k + 1):
                        hit_counters[criterion][hit_k] += 1

    accuracies = {
        criterion: {k: hits / total_reactions for k, hits in counter.items()}
        for criterion, counter in hit_counters.items()
    }

    return accuracies


def optimize_weights(
    config_path: str, val_data_path: str, n_trials: int, sample_size: int, use_top_k: int
) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load encoders
    agent_encoder, agent_standardizer, rxn_encoder, featurizer = load_encoders(cfg)

    # load validation data
    val_data = load_validation_data(val_data_path, sample_size)
    validation_dataset = EvaluationDatasetFactory.for_inference(
        data=val_data,
        agent_standardizer=agent_standardizer,
        agent_encoder=agent_encoder,
        rxn_encoder=rxn_encoder,
        featurizer=featurizer,
    )

    # load models only (weights will be optimized)
    models, model_types, _ = load_models_from_yaml(config_path, device)

    def objective(trial):
        suggested_weights = {
            "agent": trial.suggest_float("agent", 0.1, 0.5, step=0.05),
            "temperature": trial.suggest_float("temperature", 0.1, 0.5, step=0.05),
            "reactant_amount": trial.suggest_float("reactant_amount", 0.1, 0.5, step=0.05),
            "agent_amount": trial.suggest_float("agent_amount", 0.1, 0.5, step=0.05),
        }

        predictor = EnumeratedPredictor(
            agent_model=models["agent"],
            temperature_model=models["temperature"],
            reactant_amount_model=models["reactant_amount"],
            agent_amount_model=models["agent_amount"],
            model_types=model_types,
            agent_encoder=agent_encoder,
            device=device,
            weights=suggested_weights,
            use_geometric=True,
        )

        accs = calculate_topk_accuracy(
            predictor.predict_many(validation_dataset, top_k=10),
            validation_dataset,
            agent_encoder,
        )
        acc_to_use = accs["overall"][use_top_k]

        logger.info(f"Trial {trial.number} overall accs: {accs['overall']}")
        return acc_to_use

    # Run optimization
    study = optuna.create_study(study_name=f"optimize_pipeline_weights", direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best weights: {study.best_params}")
    logger.info(f"Best accuracy (top-{use_top_k}): {study.best_value:.4f}")

    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(description="Optimize stage weights")
    add_opt_opts(parser)
    args = parser.parse_args()

    # set up logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    log_file = cfg.logs_dir / "optimize_weights.log"
    logger.add(log_file, level="INFO")

    val_data_path = args.val_data or (cfg.processed_data_dir / "overlap/overlap_val.pickle")
    if not Path(val_data_path).exists():
        raise FileNotFoundError(f"no validation data found at {val_data_path}")

    # run optimization
    best_weights, best_score = optimize_weights(
        config_path=args.config_path,
        val_data_path=val_data_path,
        n_trials=args.n_trials,
        sample_size=args.sample_size,
        use_top_k=args.use_top_k,
    )
    logger.info(f"Best weights: {best_weights}")
    logger.info(f"Best accuracy (top-{args.use_top_k}): {best_score:.4f}")


if __name__ == "__main__":
    main()
