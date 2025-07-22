import sys
import logging
import pickle
import faiss
import json
import numpy as np
from loguru import logger
from typing import List, Dict
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool

from quarc.data.datapoints import ReactionDatum

RDLogger.DisableLog("rdApp.*")
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)


def generate_fingerprint(agent_records) -> np.ndarray:
    """
    Generate a combined fingerprint by OR-ing together fingerprints of each SMILES in smiles_list.
    """
    smiles_list = [a.smiles for a in agent_records]
    merged_smi = ".".join(smiles_list)
    mol = Chem.MolFromSmiles(merged_smi)
    if (not mol) or (mol.GetNumHeavyAtoms() == 0):
        raise ValueError(f"Failed to parse SMILES: {merged_smi}")
    fp_arr = fp_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)  # shape (2048,) dtype=uint8
    return fp_arr


def generate_reaction_fingerprint(reaction_datum: ReactionDatum) -> np.ndarray:
    try:
        FP_r = generate_fingerprint(reaction_datum.reactants)
        FP_p = generate_fingerprint(reaction_datum.products)
        FP_rxn_np = np.concatenate((FP_r, FP_p))
        return FP_rxn_np
    except ValueError as e:
        logger.error(
            f"Failed to generate fingerprint for rxn {reaction_datum.document_id} {reaction_datum.date}: {e}"
        )
        return np.zeros(4096, dtype=np.uint8)


def parallel_fingerprint_generation(data_chunk):
    return np.vstack([generate_reaction_fingerprint(d) for d in data_chunk])


def precompute_fingerprints(data, save_path: str):
    num_processes = 32
    chunk_size = len(data) // num_processes
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"{len(chunks)} chunks, each with {chunk_size} data points.")

    pool = Pool(processes=num_processes)
    fingerprint_chunks = pool.map(parallel_fingerprint_generation, chunks)

    train_vectors = np.vstack(fingerprint_chunks)
    print(f"Saving fingerprints to {save_path}")
    np.save(save_path, train_vectors)


def build_class_indices(
    train_data: list[ReactionDatum],
    train_vectors: np.ndarray,
    output_dir: Path,
) -> tuple[Dict[str, faiss.Index], Dict[str, List[int]]]:
    """
    Build Faiss indices per reaction class with batched processing.
    Saves intermediate results to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_train_indices = defaultdict(
        list
    )  # key: reaction_class, value: list of indices of the data points in the train_data

    logger.info("Collecting class information...")
    for i, datum in enumerate(tqdm(train_data)):
        cls = datum.rxn_class
        class_to_train_indices[cls].append(i)

    # Save class information
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_to_train_indices, f)

    class_to_index = {}  # key: reaction_class, value: faiss.IndexBinaryFlat
    for cls, indices in tqdm(class_to_train_indices.items(), desc="Building indices"):
        vectors = train_vectors[indices]
        index = faiss.IndexFlat(
            vectors.shape[1], faiss.METRIC_Jaccard
        )  # need to be float for custom metric
        index.add(vectors)
        class_to_index[cls] = index

        # save index individually
        # faiss.write_index(index, str(output_dir / f"index_{cls}.faiss"))

    with open(output_dir / "class_to_index.pkl", "wb") as f:
        pickle.dump(class_to_index, f)


def get_local_nearest_neighbors(
    test_data: list[ReactionDatum],
    test_vectors: np.ndarray,
    class_to_index: Dict[str, faiss.Index],
    output_path: Path,
    k: int = 10,
) -> None:
    """
    Find k-nearest neighbors for test reactions within their respective classes.
    Processes in batches and saves results as pickle files per class.
    """
    # Group test data by class
    class_to_test_indices = defaultdict(list)
    for i, datum in enumerate(test_data):
        class_to_test_indices[datum.rxn_class].append(i)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = [
        {"reaction_class": None, "distances": None, "neighbors": None}
        for _ in range(len(test_data))
    ]

    # Process each class
    for cls, test_indices in tqdm(class_to_test_indices.items(), desc="Processing classes"):
        if cls not in class_to_index:
            # Handle missing classes
            for test_idx in test_indices:
                results[test_idx] = {
                    "reaction_class": cls,
                    "distances": [-1] * k,
                    "neighbors": [None] * k,
                }
        else:
            test_class_vectors = test_vectors[test_indices]
            D, I = class_to_index[cls].search(test_class_vectors, k)

            # Store results for this class
            # ! D/disctances is actually Jaccard similarity
            for test_idx, distances, neighbors in zip(test_indices, D, I):
                results[test_idx] = {
                    "reaction_class": cls,
                    "distances": distances,
                    "neighbors": neighbors,
                }

    # Save all results to a single pickle file
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results written to {output_path}")
