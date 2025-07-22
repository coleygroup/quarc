import os
import pickle

import pandas as pd
from loguru import logger

max_num_reactants = 11
max_num_agents = 16


def merge_and_deduplicate(config: dict):
    logger.info("--- merge and deduplicate ---")
    temp_folder = config["data_collection"]["temp_dedup_dir"]
    output_path = config["data_collection"]["output_path"]

    temp_files = [
        os.path.join(temp_folder, file_name)
        for file_name in os.listdir(temp_folder)
        if file_name.startswith("temp_deduped_") and file_name.endswith(".pkl")
    ]
    all_deduped_records = []
    for temp_file_path in temp_files:
        print(f"Reading temp file: {os.path.basename(temp_file_path)}")
        with open(temp_file_path, "rb") as temp_file:
            deduped_records = pickle.load(temp_file)
            all_deduped_records.extend(deduped_records)
    print("Finished reading all temp files")
    print(f"Total number of records: {len(all_deduped_records)}")
    logger.info(f"Total number of records: {len(all_deduped_records)}")
    
    # pre-allocate df
    2 + (max_num_reactants * 2) + (max_num_agents * 2)  # base cols + reactant cols + agent cols

    # Set column names
    columns = ["rxn_smiles", "temperature"]
    columns.extend([f"reactant_{i}_smi" for i in range(1, max_num_reactants + 1)])
    columns.extend([f"reactant_{i}_amt" for i in range(1, max_num_reactants + 1)])
    columns.extend([f"agent_{i}_smi" for i in range(1, max_num_agents + 1)])
    columns.extend([f"agent_{i}_amt" for i in range(1, max_num_agents + 1)])

    df = pd.DataFrame(None, index=range(len(all_deduped_records)), columns=columns)

    # Fill data
    for i, datum in enumerate(all_deduped_records):
        if i % 1_000_000 == 0:
            print(f"Processed {i} records")

        # Base columns
        df.iloc[i, 0] = datum.rxn_smiles
        df.iloc[i, 1] = round(datum.temperature, 2) if datum.temperature else None

        # Fill reactants (sorted by SMILES)
        sorted_reactants = sorted(datum.reactants, key=lambda x: x.smiles)
        reactant_smi_start = 2
        reactant_amt_start = reactant_smi_start + max_num_reactants

        for j, reactant in enumerate(sorted_reactants):
            df.iloc[i, reactant_smi_start + j] = reactant.smiles
            df.iloc[i, reactant_amt_start + j] = (
                round(reactant.amount, 4) if reactant.amount else None
            )

        # Fill agents (also sorted by SMILES)
        sorted_agents = sorted(datum.agents, key=lambda x: x.smiles)
        agent_smi_start = 2 + (max_num_reactants * 2)
        agent_amt_start = agent_smi_start + max_num_agents

        for j, agent in enumerate(sorted_agents):
            df.iloc[i, agent_smi_start + j] = agent.smiles
            df.iloc[i, agent_amt_start + j] = round(agent.amount, 4) if agent.amount else None

    # Perform final deduplication
    df.drop_duplicates(inplace=True)

    # Deduplicate with DataFrame,
    deduped_idx = df.index.values
    deduped_records = [all_deduped_records[i] for i in deduped_idx]
    print(f"Total number of records after deduplication: {len(deduped_records)}")
    logger.info(f"Total number of records after deduplication: {len(deduped_records)}")

    with open(output_path, "wb") as f:
        pickle.dump(deduped_records, f, pickle.HIGHEST_PROTOCOL)
