import pickle
from pathlib import Path
from quarc.predictors.base import BasePredictor, StagePrediction, PredictionList
from quarc.data.eval_datasets import ReactionInput
from quarc.baselines.popularity_baseline import CanonicalCondition


class PopularityPredictor(BasePredictor):
    """
    Popularity baseline predictor using precomputed top-k conditions per reaction class.
    Only handles prediction generation - no binning or evaluation logic.
    """

    def __init__(self, popularity_data_path: Path):
        """
        Args:
            popularity_data_path: top 10 overall conditions
                dict[rxn_class] -> list[(CanonicalCondition, count)]
            top_k: Maximum number of predictions to return
        """

        self._load_popularity_data(popularity_data_path)

    def _load_popularity_data(self, popularity_data_path: Path):
        with open(popularity_data_path, "rb") as f:
            self.popularity_data = pickle.load(f)

    def predict(self, reaction: ReactionInput, top_k: int = 10) -> PredictionList:
        rxn_class = reaction.metadata["rxn_class"]
        doc_id = reaction.metadata["doc_id"]
        rxn_smiles = reaction.metadata["rxn_smiles"]

        predictions = []

        if rxn_class not in self.popularity_data:
            return PredictionList(
                doc_id=doc_id, rxn_class=rxn_class, rxn_smiles=rxn_smiles, predictions=[]
            )

        popular_conditions = self.popularity_data[rxn_class][:top_k]

        total_count = sum(count for _, count in popular_conditions)

        for condition, count in popular_conditions:
            agents = [agent_idx for agent_idx, _ in condition.binned_agents]

            temp_bin = condition.binned_temperature

            reactant_bins = list(condition.binned_reactant_ratios)

            agent_amount_bins = [
                (agent_idx, bin_idx) for agent_idx, bin_idx in condition.binned_agents
            ]

            score = float(count) / total_count if total_count > 0 else 0.0

            stage_pred = StagePrediction(
                agents=agents,
                temp_bin=temp_bin,
                reactant_bins=reactant_bins,
                agent_amount_bins=agent_amount_bins,
                score=score,
                meta={"condition_count": count, "total_count": total_count},
            )

            predictions.append(stage_pred)

        return PredictionList(
            doc_id=doc_id, rxn_class=rxn_class, rxn_smiles=rxn_smiles, predictions=predictions
        )
