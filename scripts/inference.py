import argparse
import json
import torch
from typing import Any
import warnings

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


def format_predictions(predictions, agent_encoder: AgentEncoder, top_k: int = 5) -> dict[str, Any]:
    binning_config = BinningConfig.default()
    temp_labels = binning_config.get_bin_labels("temperature")
    reactant_labels = binning_config.get_bin_labels("reactant")
    agent_labels = binning_config.get_bin_labels("agent")

    results = {
        "doc_id": predictions.doc_id,
        "rxn_class": predictions.rxn_class,
        "rxn_smiles": predictions.rxn_smiles,
        "predictions": [],
    }
    reactants_smiles, _, _ = parse_rxn_smiles(predictions.rxn_smiles)

    for i, pred in enumerate(predictions.predictions[:top_k]):
        agent_smiles = agent_encoder.decode(pred.agents)
        temp_label = temp_labels[pred.temp_bin]
        reactant_labels_list = [reactant_labels[bin_idx] for bin_idx in pred.reactant_bins]

        agent_amounts = []
        for agent_idx, bin_idx in pred.agent_amount_bins:
            agent_smi = agent_encoder.decode([agent_idx])[0]
            amount_label = agent_labels[bin_idx]
            agent_amounts.append({"agent": agent_smi, "amount_range": amount_label})

        reactant_amounts = []
        for reactant_smi, reactant_label in zip(reactants_smiles, reactant_labels_list):
            reactant_amounts.append({"reactant": reactant_smi, "amount_range": reactant_label})

        prediction = {
            "rank": i + 1,
            "score": pred.score,
            "agents": agent_smiles,
            "temperature": temp_label,
            "reactant_amounts": reactant_amounts,
            "agent_amounts": agent_amounts,
            "raw_scores": pred.meta if hasattr(pred, "meta") else {},
        }
        results["predictions"].append(prediction)

    return results


def main():
    parser = argparse.ArgumentParser(description="QUARC Inference")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with reactions")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for predictions")
    parser.add_argument(
        "--config",
        "-c",
        default="ffn_pipeline.yaml",
        help="Pipeline config (ffn_pipeline.yaml or gnn_pipeline.yaml)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of top predictions to return"
    )
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")

    args = parser.parse_args()

    cfg = load_settings()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load supporting components
    a_enc = AgentEncoder(
        class_path=cfg.processed_data_dir / "agent_encoder/agent_encoder_list.json"
    )
    a_standardizer = AgentStandardizer(
        conv_rules=cfg.processed_data_dir / "agent_encoder/agent_rules_v1.json",
        other_dict=cfg.processed_data_dir / "agent_encoder/agent_other_dict.json",
    )
    rxn_encoder = ReactionClassEncoder(class_path=cfg.pistachio_namerxn_path)
    featurizer = CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")

    # Load models
    models, model_types, weights = load_models_from_yaml(cfg.checkpoints_dir / args.config, device)

    # Setup predictor
    predictor = EnumeratedPredictor(
        agent_model=models["agent"],
        temperature_model=models["temperature"],
        reactant_amount_model=models["reactant_amount"],
        agent_amount_model=models["agent_amount"],
        model_types=model_types,
        agent_encoder=a_enc,
        device=device,
        weights=weights["use_top_5"],
        use_geometric=weights["use_geometric"],
    )

    # Load and convert input to ReactionDatum objects
    with open(args.input, "r") as f:
        input_data = json.load(f)

    reactions = []
    for i, item in enumerate(input_data):
        rxn_smiles = item["rxn_smiles"]
        rxn_class = item["rxn_class"]
        doc_id = item.get("doc_id", f"reaction_{i}")

        reactants, agents, products = parse_rxn_smiles(rxn_smiles)

        reactions.append(
            ReactionDatum(
                rxn_smiles=rxn_smiles,
                reactants=[AgentRecord(smiles=r, amount=None) for r in reactants],
                agents=[AgentRecord(smiles=a, amount=None) for a in agents],
                products=[AgentRecord(smiles=p, amount=None) for p in products],
                rxn_class=rxn_class,
                document_id=doc_id,
                date=None,
                temperature=None,
            )
        )

    dataset = EvaluationDatasetFactory.for_inference(
        data=reactions,
        agent_standardizer=a_standardizer,
        agent_encoder=a_enc,
        rxn_encoder=rxn_encoder,
        featurizer=featurizer,
    )

    all_results = []
    for reaction in dataset:
        predictions = predictor.predict(reaction, top_k=args.top_k)
        result = format_predictions(predictions, a_enc, top_k=args.top_k)
        all_results.append(result)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
