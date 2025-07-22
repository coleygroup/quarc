import json
import pickle
import os
from collections import Counter

import pandas as pd
from loguru import logger

from quarc.utils.smiles_utils import get_common_solvents_canonical

COMMON_SOLVENTS_CANONICAL = get_common_solvents_canonical()

INVALID_SMILES = [
    "null",
    "S=[NH4+]",
    "C[ClH]",
    "[OH2-]",
    "[ClH-]",
    "[P+3]",
    "[B]",
]


def count_agent_distribution(config) -> Counter:
    """Count agent distribution from agents with amounts"""

    input_path = config["generate_agent_class"]["input_path"]
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    agent_distribution = Counter()
    for d in data:
        for agent in d.agents:
            if agent.amount is not None:
                agent_distribution[agent.smiles] += 1

    return agent_distribution


def generate_agent_class(config):
    """Generate agent class. Config file contains the list of elements needed to be saved to others, minocc

    1. Agent frequency (counted with amount) >= minocc;
    2. metal containing agent frequency >= metal_minocc;
    3. Append not included common solvents;
    3. Other_dict:all other metal containing agent mapped to "other_{Pd/Rh/...};
    """
    logger.info("---Generating agent class---")
    minocc = config["generate_agent_class"]["minocc"]
    metal_minocc = config["generate_agent_class"]["metal_minocc"]
    element_list = ["Pd", "Rh", "Pt", "Ni", "Ru", "Ir", "Fe", "Ag"]

    agent_distribution = count_agent_distribution(config)

    conv_rules_path = os.path.join(os.path.dirname(__file__), "../utils/agent_rules_v1.json")
    # load conv_rules
    with open(conv_rules_path, "r") as f:
        conv_rules = json.load(f)

    # apply conv_rules
    df = pd.DataFrame.from_dict(
        agent_distribution, orient="index", columns=["occurrence"]
    ).reset_index()
    df["standardized"] = df["index"].apply(lambda x: conv_rules.get(x, x))

    # aggregate by standardized smiles
    df_agg = df.groupby("standardized").agg({"occurrence": "sum"}).reset_index()
    df_agg.sort_values(by="occurrence", ascending=False, inplace=True)

    # extract with given minocc - base
    df_agg_minocc = df_agg[(df_agg["occurrence"] >= minocc)]
    encoder_list_minocc = df_agg_minocc["standardized"].tolist()
    logger.info(f"len(encoder_list_minocc{minocc}): {len(encoder_list_minocc)}")

    # curate "other" dictionary and lower cutoff
    df_agg_rare = df_agg[(df_agg["occurrence"] < minocc)]

    encoder_lists = []
    encoder_other_lists = []
    for element in element_list:
        if element == "Fe":
            temp_df = df_agg_rare[
                df_agg_rare["standardized"].str.contains("Fe")
                & ~df_agg_rare["standardized"].str.contains("Pd")
            ]
        else:
            temp_df = df_agg_rare[df_agg_rare["standardized"].str.contains(element)]

        # 1) lower the cut-off to 50
        temp_encoder_list = temp_df[temp_df["occurrence"] >= metal_minocc]["standardized"].tolist()
        encoder_lists.append(temp_encoder_list)
        logger.info(f"len(encoder_list_{element}_50): {len(temp_encoder_list)}")

        # 2) create "other" list
        temp_other_list = temp_df[temp_df["occurrence"] < metal_minocc]["standardized"].tolist()
        encoder_other_lists.append(temp_other_list)
        logger.info(f"len(encoder_other_{element}): {len(temp_other_list)}")

    # if overlap in other-list (eg. '[Pd].[Pt]'), just remove
    # iterate over each encoder list and compare with all other encoder lists
    for i, encoder_list in enumerate(encoder_lists):
        for j in range(i + 1, len(encoder_lists)):
            overlap = set(encoder_list).intersection(set(encoder_lists[j]))
            if overlap:
                logger.debug(f"Overlap between {element_list[i]} and {element_list[j]}: {overlap}")

    # iterate over each encoder_other list and compare with all other encoder_other lists
    all_overlaps = []
    for i, encoder_other_list in enumerate(encoder_other_lists):
        for j in range(i + 1, len(encoder_other_lists)):
            overlap = set(encoder_other_list).intersection(set(encoder_other_lists[j]))
            if overlap:
                all_overlaps.extend(list(overlap))
                logger.debug(
                    f"Overlap other between {element_list[i]} and {element_list[j]}: {overlap}"
                )

    # curate the final agent encoder
    final_encoder_list = []
    for encoder_list in encoder_lists:
        final_encoder_list.extend(encoder_list)
    final_encoder_list.extend(encoder_list_minocc)
    logger.debug(f"len(final_encoder_list) before: {len(final_encoder_list)}")

    # filter out INVALID_SMILES
    final_encoder_list = [smiles for smiles in final_encoder_list if smiles not in INVALID_SMILES]
    logger.debug(f"len(final_encoder_list) after remove invalid smiles: {len(final_encoder_list)}")

    # save the other dictionary that maps additional smi to other label
    other_dict = {}
    for i, encoder_other_list in enumerate(encoder_other_lists):
        # Add the other label to final encoder list
        label = f"other_{element_list[i]}"
        final_encoder_list.append(label)

        for item in encoder_other_list:
            if item not in all_overlaps:
                other_dict[item] = label

    logger.info(f"len(other_dict): {len(other_dict)}")
    logger.info(f"len(final_encoder_list) after other: {len(final_encoder_list)}")

    # Append not included common solvents
    not_included_common_solvents = list(COMMON_SOLVENTS_CANONICAL - set(final_encoder_list))
    logger.info(f"len(not_included_common_solvents): {len(not_included_common_solvents)}")
    final_encoder_list.extend(not_included_common_solvents)

    # save the final encoder list
    output_encoder_path = config["generate_agent_class"]["output_encoder_path"]
    with open(output_encoder_path, "w") as f:
        json.dump(final_encoder_list, f, indent=4)

    output_other_dict_path = config["generate_agent_class"]["output_other_dict_path"]
    with open(output_other_dict_path, "w") as f:
        json.dump(other_dict, f, indent=4)
