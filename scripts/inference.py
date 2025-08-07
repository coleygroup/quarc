import argparse
import json
import pickle
import torch
from typing import Any
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from chemprop.featurizers import CondensedGraphOfReactionFeaturizer

from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.models.modules.rxn_encoder import ReactionClassEncoder
from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.data.eval_datasets import EvaluationDatasetFactory
from quarc.data.binning import BinningConfig
from quarc.utils.smiles_utils import parse_rxn_smiles
from quarc.predictors.model_factory import load_models_from_yaml
from quarc.predictors.multistage_predictor import EnumeratedPredictor
from quarc.settings import load as load_settings
from quarc.cli.quarc_parser import add_predict_opts


def convert_json_to_reactions(data: list[dict]) -> list[ReactionDatum]:
    """Convert JSON input to ReactionDatum objects"""

    reactions = []
    for i, item in enumerate(data):
        reactants, agents, products = parse_rxn_smiles(item["rxn_smiles"])

        reactions.append(
            ReactionDatum(
                rxn_smiles=item["rxn_smiles"],
                reactants=[AgentRecord(smiles=r, amount=None) for r in reactants],
                agents=[AgentRecord(smiles=a, amount=None) for a in agents],
                products=[AgentRecord(smiles=p, amount=None) for p in products],
                rxn_class=item["rxn_class"],
                document_id=item.get("doc_id", f"reaction_{i}"),
                date=None,
                temperature=None,
            )
        )

    return reactions


def format_predictions(predictions, agent_encoder: AgentEncoder) -> dict[str, Any]:
    """Format prediction results for JSON output"""
    binning_config = BinningConfig.default()
    temp_labels = binning_config.get_bin_labels("temperature")
    reactant_labels = binning_config.get_bin_labels("reactant")
    agent_labels = binning_config.get_bin_labels("agent")

    reactants_smiles, _, _ = parse_rxn_smiles(predictions.rxn_smiles)

    formatted_predictions = []
    for i, pred in enumerate(predictions.predictions):
        agent_amounts = [
            {"agent": agent_encoder.decode([agent_idx])[0], "amount_range": agent_labels[bin_idx]}
            for agent_idx, bin_idx in pred.agent_amount_bins
        ]

        reactant_amounts = [
            {"reactant": reactant_smi, "amount_range": reactant_labels[bin_idx]}
            for reactant_smi, bin_idx in zip(reactants_smiles, pred.reactant_bins)
        ]

        formatted_predictions.append(
            {
                "rank": i + 1,
                "score": pred.score,
                "agents": agent_encoder.decode(pred.agents),
                "temperature": temp_labels[pred.temp_bin],
                "reactant_amounts": reactant_amounts,
                "agent_amounts": agent_amounts,
                "raw_scores": getattr(pred, "meta", {}),
            }
        )

    return {
        "doc_id": predictions.doc_id,
        "rxn_class": predictions.rxn_class,
        "rxn_smiles": predictions.rxn_smiles,
        "predictions": formatted_predictions,
    }


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


def load_data(input_path: str) -> list[ReactionDatum]:
    # json of just smiles + rxn class
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
        reactions = convert_json_to_reactions(data)

    # e.g., overlap_val already in ReactionDatum format
    elif input_path.endswith(".pickle") or input_path.endswith(".pkl"):
        with open(input_path, "rb") as f:
            reactions = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")

    return reactions


def run_inference(input_path: str, output_path: str, config_path: str, top_k: int):
    """Run complete inference pipeline"""
    cfg = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load encoders
    agent_encoder, agent_standardizer, rxn_encoder, featurizer = load_encoders(cfg)

    # load models
    models, model_types, weights = load_models_from_yaml(config_path, device)

    # load predictor
    predictor = EnumeratedPredictor(
        agent_model=models["agent"],
        temperature_model=models["temperature"],
        reactant_amount_model=models["reactant_amount"],
        agent_amount_model=models["agent_amount"],
        model_types=model_types,
        agent_encoder=agent_encoder,
        device=device,
        weights=weights["use_top_5"],
        use_geometric=weights["use_geometric"],
    )

    # load data
    reactions = load_data(input_path)
    dataset = EvaluationDatasetFactory.for_inference(
        data=reactions,
        agent_standardizer=agent_standardizer,
        agent_encoder=agent_encoder,
        rxn_encoder=rxn_encoder,
        featurizer=featurizer,
    )

    # run inference
    results = [
        format_predictions(predictor.predict(r, top_k=top_k), agent_encoder) for r in dataset
    ]

    # save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="QUARC Inference")
    add_predict_opts(parser)

    args, unknown = parser.parse_known_args()

    run_inference(args.input, args.output, args.config_path, args.top_k)


if __name__ == "__main__":
    main()
