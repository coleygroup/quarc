import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger


def split_by_document(config: dict) -> tuple[list, list, list]:
    """Split data into train/val/test sets by document ID while maintaining approximate reaction ratios"""
    logger.info("---Splitting by document---")
    document_split_config = config["document_split"]
    input_path = document_split_config["input_path"]
    output_dir = document_split_config["output_dir"]
    split_ratios = document_split_config["split_ratios"]
    seed = document_split_config["seed"]
    save_split_info = document_split_config["save_split_info"]

    # Read data
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Get document counts
    document_counts = Counter(d.document_id.split("_")[0] for d in data)
    initial_docs = len(document_counts)
    initial_rxns = len(data)

    # Get split ratios
    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]

    # Group documents by size ranges for more balanced splits
    size_ranges = [
        (1, 10),
        (11, 20),
        (21, 30),
        (31, 40),
        (41, 60),
        (61, 100),
        (101, 200),
        (201, 300),
        (301, 500),
        (501, 1000),
        (1000, 3000),
    ]

    grouped_docs = defaultdict(list)
    for doc_id, count in document_counts.items():
        for lower, upper in size_ranges:
            if lower <= count <= upper:
                grouped_docs[(lower, upper)].append(doc_id)
                break

    # Split documents within each size range
    random.seed(seed)
    train_docs, val_docs, test_docs = set(), set(), set()

    for _size_range, docs in grouped_docs.items():
        random.shuffle(docs)
        n_docs = len(docs)
        n_train = int(train_ratio * n_docs)
        n_val = int(val_ratio * n_docs)

        train_docs.update(docs[:n_train])
        val_docs.update(docs[n_train : n_train + n_val])
        test_docs.update(docs[n_train + n_val :])

    # Split data based on document assignments
    train_data, val_data, test_data = [], [], []
    for datum in data:
        doc_id = datum.document_id.split("_")[0]
        if doc_id in train_docs:
            train_data.append(datum)
        elif doc_id in val_docs:
            val_data.append(datum)
        else:
            test_data.append(datum)

    # Log split statistics
    logger.info(
        f"Document Split Statistics:\n"
        f"seed: {seed}\n"
        f"Overall: {initial_docs} total documents, {initial_rxns} total reactions\n"
        f"Train: {len(train_docs)} docs, {len(train_data)} rxns ({len(train_data)/initial_rxns:.1%})\n"
        f"Val: {len(val_docs)} docs, {len(val_data)} rxns ({len(val_data)/initial_rxns:.1%})\n"
        f"Test: {len(test_docs)} docs, {len(test_data)} rxns ({len(test_data)/initial_rxns:.1%})"
    )

    # Save split information if requested
    if save_split_info:
        detailed_documents = {
            "train_docs": list(train_docs),
            "val_docs": list(val_docs),
            "test_docs": list(test_docs),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / "train_val_test_document_ids.json", "w") as f:
            json.dump(detailed_documents, f, indent=2)

    # save train, val, test
    with open(output_dir / "full_train.pickle", "wb") as f:
        pickle.dump(train_data, f)
    with open(output_dir / "full_val.pickle", "wb") as f:
        pickle.dump(val_data, f)
    with open(output_dir / "full_test.pickle", "wb") as f:
        pickle.dump(test_data, f)
