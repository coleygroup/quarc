import pickle
import json
import numpy as np
from pathlib import Path
import pandas as pd
from quarc.data.datapoints import ReactionDatum
from quarc.data.eval_datasets import ReactionInput
from quarc.predictors.base import BasePredictor, PredictionList, StagePrediction


class NearestNeighborPredictor(BasePredictor):
    """
    Nearest neighbor baseline predictor using precomputed neighbor indices.
    Uses experimental conditions from nearest training reactions as predictions.
    """

    def __init__(
        self,
        enhanced_neighbors_path: Path,
        train_data: list[ReactionDatum],
        agent_encoder,
        agent_standardizer,
        binning_config,
    ):
        """
        Args:
            enhanced_neighbors_path: Path to enhanced neighbors data with identifiers
            train_data: Training dataset for neighbor lookup
            agent_encoder: Agent encoder for converting SMILES to indices
            agent_standardizer: Agent standardizer
            binning_config: Binning configuration for digitizing continuous values
            top_k: Maximum number of neighbor predictions to return
        """
        self.train_data = train_data
        self.agent_encoder = agent_encoder
        self.agent_standardizer = agent_standardizer
        self.binning_config = binning_config

        self.neighbors_df = pd.read_pickle(enhanced_neighbors_path)

        self._create_lookup_index()

    def _create_lookup_index(self):
        self.lookup_index = {}

        for idx, row in self.neighbors_df.iterrows():
            doc_id = row["document_id"]
            rxn_smiles = row["rxn_smiles"]

            key = (doc_id, rxn_smiles)
            self.lookup_index[key] = idx

    def predict(self, reaction: ReactionInput, top_k: int = 10) -> PredictionList:

        doc_id = reaction.metadata["doc_id"]
        rxn_class = reaction.metadata["rxn_class"]
        rxn_smiles = reaction.metadata["rxn_smiles"]

        predictions = []

        # Use precomputed look up index to get global indices of neighbors
        key = (doc_id, rxn_smiles)
        neighbor_row_idx = self.lookup_index.get(key, None)

        if neighbor_row_idx is None:
            return PredictionList(
                doc_id=doc_id, rxn_class=rxn_class, rxn_smiles=rxn_smiles, predictions=[]
            )

        row = self.neighbors_df.iloc[neighbor_row_idx]
        neighbors = row["global_neighbors"]
        distances = row["distances"]

        if neighbors is None:
            return PredictionList(
                doc_id=doc_id, rxn_class=rxn_class, rxn_smiles=rxn_smiles, predictions=[]
            )

        for i, (neighbor_idx, distance) in enumerate(zip(neighbors[:top_k], distances[:top_k])):
            neighbor_reaction = self.train_data[neighbor_idx]

            stage_pred = self._extract_conditions(neighbor_reaction, distance, neighbor_idx)
            if stage_pred is not None:
                predictions.append(stage_pred)

        return PredictionList(
            doc_id=doc_id, rxn_class=rxn_class, rxn_smiles=rxn_smiles, predictions=predictions
        )

    def _extract_conditions(
        self,
        neighbor_reaction: ReactionDatum,
        distance: float,
        neighbor_idx: int,
    ) -> StagePrediction:
        """
        Extract experimental conditions from a neighbor reaction and format as StagePrediction.

        Args:
            neighbor_reaction: Training reaction data object
            distance: Actually is Tanimoto/Jaccard similarity to this neighbor (not distance)

        Returns:
            StagePrediction with neighbor's conditions, or None if missing data
        """
        try:
            if (
                neighbor_reaction.agents is None
                or neighbor_reaction.reactants is None
                or neighbor_reaction.temperature is None
                or any(r.amount is None for r in neighbor_reaction.reactants)
                or any(a.amount is None for a in neighbor_reaction.agents)
            ):
                return None

            # stage 1
            standardized_smiles = self.agent_standardizer.standardize(
                [agent.smiles for agent in neighbor_reaction.agents]
            )
            agents = self.agent_encoder.encode(standardized_smiles)

            # stage 2
            temp_bin = np.digitize(
                neighbor_reaction.temperature, self.binning_config.temperature_bins
            )

            # stage 3
            reactant_amounts = [r.amount for r in neighbor_reaction.reactants]
            limiting_amount = min(reactant_amounts)
            normalized_amounts = [amt / limiting_amount for amt in reactant_amounts[:5]]
            reactant_bins = list(
                np.digitize(normalized_amounts, self.binning_config.reactant_amount_bins)
            )

            # stage 4
            agent_amount_bins = []
            for agent in neighbor_reaction.agents:
                agent_smiles = agent.smiles
                agent_amount = agent.amount

                standardized_smiles = self.agent_standardizer.standardize([agent_smiles])
                agent_idx = self.agent_encoder.encode(standardized_smiles)[0]
                relative_amount = agent_amount / limiting_amount
                bin_idx = np.digitize([relative_amount], self.binning_config.agent_amount_bins)[0]
                agent_amount_bins.append((agent_idx, bin_idx))

            agent_amount_bins.sort(key=lambda x: x[0])

            score = distance  # actually is tanimoto/jaccard similarity

            return StagePrediction(
                agents=agents,
                temp_bin=temp_bin,
                reactant_bins=reactant_bins,
                agent_amount_bins=agent_amount_bins,
                score=score,
                meta={
                    "neighbor_doc_id": neighbor_reaction.document_id,
                    "neighbor_idx": neighbor_idx,
                },
            )

        except Exception as e:
            raise ValueError(f"Error processing neighbor reaction: {e}")
