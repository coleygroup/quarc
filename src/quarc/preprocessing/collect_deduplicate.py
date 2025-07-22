""" Collect pistachio reaction record from JSON to ReactionDatum. This process involves reaction information
extraction, reaction parsing, and reaction matching, and deduplication at reaction level.
"""

import json
import os
import pickle
from multiprocessing import Pool

import pandas as pd
from loguru import logger
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from quarc.data.datapoints import AgentRecord, ReactionDatum
from quarc.preprocessing.exceptions import *
from quarc.preprocessing.preprocess_quantities import (
    determine_category,
    process_acidbase,
    process_exceptions,
    process_mixuture,
    process_simple,
)
from quarc.preprocessing.text_parser import parse_temperature
from quarc.utils.smiles_utils import (
    canonicalize_smiles,
    is_atom_map_rxn_smiles,
    remove_stereo,
)


def get_temperature(r):
    """Extract temperature, limited to one temperature per reaction"""
    for a in r["actions"]:
        if a.get("parameters") is not None:
            for p in a["parameters"]:
                if p["type"] == "Temperature":
                    try:
                        parsed_t = parse_temperature(p["text"])
                    except:
                        parsed_t = None
                    return parsed_t
    return None


def getq(qs, s):
    for i in qs:
        if i["type"] == s:
            return i
    return None


def merge_duplicate_smiles(components):
    """
    Rules for merging:
        If Molarity is given, different molarity treated as different compound
        Otherwise, merge quantities, sum up fields
    """
    res = []
    for c in components:
        found_in_res = False
        if c.get("smiles") is None:
            res.append(c)
            continue
        c["smiles"] = canonicalize_smiles(c["smiles"].split(" ")[0])
        idx_res = None
        for i in range(len(res)):
            if res[i].get("smiles") is None:
                continue
            if res[i]["smiles"] == c["smiles"]:
                found_in_res = True
                idx_res = i
                break
        if not found_in_res:
            res.append(c)
        else:
            # try to merge "quantities" field
            if c.get("quantities") is not None:
                if res[idx_res].get("quantities") is None:
                    res[idx_res]["quantities"] = c["quantities"]
                else:
                    # different Molarity?
                    res_mol = getq(res[idx_res]["quantities"], "Molarity")
                    c_mol = getq(c["quantities"], "Molarity")
                    if (res_mol is None and c_mol is None) or (
                        res_mol is not None and c_mol is not None
                    ):
                        for c_q in c["quantities"]:
                            if c_q["type"] == "Molarity":
                                continue
                            found_same_type = False
                            for i in range(len(res[idx_res]["quantities"])):
                                if c_q["type"] == res[idx_res]["quantities"][i]["type"]:
                                    found_same_type = True
                                    if (
                                        res[idx_res]["quantities"][i].get("value")
                                        == "NaN"
                                        or res[idx_res]["quantities"][i].get("value")
                                        is None
                                        or c_q.get("value") == "NaN"
                                        or c_q.get("value") is None
                                    ):
                                        res[idx_res]["quantities"][i]["value"] = None
                                    else:
                                        res[idx_res]["quantities"][i]["value"] += c_q[
                                            "value"
                                        ]
                                    res[idx_res]["quantities"][i][
                                        "text"
                                    ] = "should not read"
                            if not found_same_type:
                                res[idx_res]["quantities"].append(c_q)
                    else:
                        res.append(c)
    return res


def create_agent_record(smiles: str, mole: float) -> AgentRecord:
    return AgentRecord(smiles, mole)


def parse_rxn_smiles(rxn_smiles: str) -> tuple[list[str], list[str], list[str]]:
    """Split rxn_smiles into reactants, agents, and products by dot, no combination needed"""
    try:
        reactants, agents, products = rxn_smiles.split(">")
        if " |" in products:
            products, _ = products.split(" |")
    except (TypeError, ValueError):
        raise BadRecord(f"\tError splitting rxn SMILES: {rxn_smiles}")

    try:
        reactants = [canonicalize_smiles(s) for s in reactants.split(".")]
        agents = [canonicalize_smiles(s) for s in agents.split(".")]
        products = [canonicalize_smiles(s) for s in products.split(".")]
    except:
        raise BadRecord(f"\tError canonicalize SMILES for {rxn_smiles}")

    reactants = [s for s in reactants if s]
    agents = [s for s in agents if s]
    products = [s for s in products if s]

    return reactants, agents, products


def forward_match(
    intermed_amounts: list[dict],
    reactant_set: set[str],
    agent_set: set[str],
    product_set: set[str],
) -> tuple[list[AgentRecord], list[AgentRecord], list[AgentRecord], list[dict], bool]:
    """Loop through component extracted intermediate quantities, check if each component is subset of parsed smiles

    Returns:
    -------
    matched_reactants: list[AgentRecord]
    matched_agents: list[AgentRecord]
    matched_products: list[AgentRecord]
    Unmatched: list[dict], list of unmatched intermed_amounts
    need_remove_stereo: bool, whether need to remove stereo from smiles
    """
    matched_reactants = []
    matched_agents = []
    matched_products = []
    need_remove_stereo = False
    unmatched = []

    for intermed_amount in intermed_amounts:
        comp_parts = set(intermed_amount["smiles"].split("."))
        role = intermed_amount["role"]
        agent_record = AgentRecord(intermed_amount["smiles"], intermed_amount["mole"])

        # Matched: Add to matched component to reactants, products, agents
        if role == "Reactant" and comp_parts.issubset(reactant_set):
            matched_reactants.append(agent_record)
            continue
        elif role == "Product" and comp_parts.issubset(product_set):
            matched_products.append(agent_record)
            continue
        elif comp_parts.issubset(agent_set):  # {Catalyst, Solvent, Agent} -> Agents
            matched_agents.append(agent_record)
            continue

        # No Matches found:
        # Exception 1: if catalyst, add to agent
        if role == "Catalyst":  # or role == 'Solvent':
            matched_agents.append(agent_record)
            continue

        # Exception 1.5: if solvent, but actually participating in reaction, add to reactant
        if role == "Solvent" and comp_parts.issubset(reactant_set):
            matched_reactants.append(agent_record)
            continue

        # Exception 2: if reactant not found, remove stereo and check again
        elif role == "Reactant":
            comp_parts_nostereo = {remove_stereo(p) for p in comp_parts}
            reactant_set_nostreo = {remove_stereo(p) for p in reactant_set}
            if comp_parts_nostereo.issubset(reactant_set_nostreo):
                need_remove_stereo = True
                matched_reactants.append(agent_record)
                continue

        # Exception 3: if product not found, remove stereo and check again
        elif role == "Product":
            comp_parts_nostereo = {remove_stereo(p) for p in comp_parts}
            product_set_nostreo = {remove_stereo(p) for p in product_set}
            if comp_parts_nostereo.issubset(product_set_nostreo):
                need_remove_stereo = True
                matched_products.append(agent_record)
                continue

        # if no matched or covered in exception, don't add
        unmatched.append(intermed_amount)

    return (
        matched_reactants,
        matched_agents,
        matched_products,
        unmatched,
        need_remove_stereo,
    )


def reverse_match(
    matched_agents: list[AgentRecord],
    matched_reactants: list[AgentRecord],
    matched_products: list[AgentRecord],
    reactant_set: set[str],
    agent_set: set[str],
    product_set: set[str],
    need_remove_stereo: bool,
) -> None:
    """Performs a reverse check to determine any unprocessed parts from parsed sets.
    Raises MismatchError if any unprocessed parts found.
    """

    processed_set = set()
    for agent in matched_agents + matched_reactants + matched_products:
        for smi in agent.smiles.split("."):
            processed_set.add(smi)

    if need_remove_stereo:
        reactant_set = {remove_stereo(p) for p in reactant_set}
        product_set = {remove_stereo(p) for p in product_set}
        processed_set = {remove_stereo(p) for p in processed_set}

    unprocessed_reactants = list(reactant_set - processed_set)
    unprocessed_products = list(product_set - processed_set)
    unprocessed_agents = list(agent_set - processed_set)

    if any([unprocessed_reactants, unprocessed_agents, unprocessed_products]):
        data = []  # list of dict
        if unprocessed_reactants:
            data.append(
                {"role": "Reactant", "unprocessed_content": unprocessed_reactants}
            )
        if unprocessed_agents:
            data.append({"role": "Agent", "unprocessed_content": unprocessed_agents})
        if unprocessed_products:
            data.append(
                {"role": "Product", "unprocessed_content": unprocessed_products}
            )
        raise ReverseMismatchError("\tUnprocessed parts found in", data=data)


def collect_reaction(r: dict) -> ReactionDatum:
    """ "collect from one raw pistachio entry and return one Reaction Datum"""

    # Check atom mapping and remove unmapped parts
    try:
        rxn_smiles = r["data"]["smiles"]
    except:
        raise BadRecord(
            f"\tEntry missing reaction SMILES for document: {r.get('title')}"
        )

    try:
        if not is_atom_map_rxn_smiles(rxn_smiles):
            raise BadRecord(f"\treaction SMILES not mapped:{rxn_smiles}")
        # FIXME: Remove un-atompped part directly causing issues - need more careful considerations
        # rxn_smiles = smiles_util.remove_unmapped_in_mapped_rxn(rxn_smiles)
    except:
        raise BadRecord(f"\tCan't check atom mapping: {rxn_smiles}")

    # Extract temperature
    temperature = get_temperature(r) if r.get("actions") else None

    # Extract quantities from components
    intermed_amounts = []
    components = r.get("components")
    components_clean = merge_duplicate_smiles(components) if components else []

    for c in components_clean:
        compo_name = c.get("smiles")
        if not compo_name:
            continue
        category = determine_category(c)
        try:
            if category == "Simple":
                intermed_amounts.extend(process_simple(c))
            elif category == "HasDotOnly":
                intermed_amounts.extend(process_exceptions(c))
            elif category == "HasDotAndNested":
                intermed_amounts.extend(process_mixuture(c))
            elif category == "HasNestedOnly":
                intermed_amounts.extend(process_acidbase(c))
        except (QuantityError, ValueError) as e:
            logger.debug(f"QuantityError in rxn_smi: {rxn_smiles}")
            logger.debug(f"Error in '{category}' for component '{compo_name}': {e}")
            if hasattr(e, "data") and e.data is not None:
                logger.debug(f"QuantityError data:{e.data}")

            # Still add to intermed_amounts, but with quantity=None to avoid ReverseMismatchError
            intermed_amounts.append(
                {
                    "smiles": compo_name,
                    "mole": None,
                    "orig_smiles": compo_name,
                    "role": c.get("role"),
                }
            )

    # Parse from rxn_smiles
    reactants, agents, products = parse_rxn_smiles(rxn_smiles)

    reactant_set = set(reactants)
    agent_set = set(agents)
    product_set = set(products)

    # Reaction must have reactants and products
    if (not reactants) or (not products):
        raise BadRecord(f"\tMissing reactants or products for {rxn_smiles}")

    # Reaction must DO something
    if product_set.issubset(reactant_set) or product_set.issubset(agent_set):
        raise BadRecord(f"\tIneffectual reaction: {rxn_smiles}")

    # Overlap between reactant_set and agent_set:
    if reactant_set.intersection(agent_set):
        raise SmilesError(
            f"\tOverlap between reactants and agents: {reactant_set.intersection(agent_set)} \n\tfor reaction: {rxn_smiles}"
        )

    # Forward matching: check if component is subset of reactant, product, agent sets.
    (
        matched_reactants,
        matched_agents,
        matched_products,
        unmatched,
        need_remove_stereo,
    ) = forward_match(intermed_amounts, reactant_set, agent_set, product_set)
    if len(unmatched) > 0:
        logger.debug(f"ForwardMismatch in rxn: {rxn_smiles}")
        logger.debug(f"unmatched:{unmatched}")

    # Reverse matching: check if any of the parsed set is not covered by components
    reverse_match(
        matched_agents,
        matched_reactants,
        matched_products,
        reactant_set,
        agent_set,
        product_set,
        need_remove_stereo,
    )  # ReverseMismatchError raised if doesn't pass

    # Jiannan: Add anything left in intermed_amounts to agents. Decide not to implement this for now.

    # Create ReactionDatum object
    datum = ReactionDatum(
        document_id=r.get("title"),
        rxn_class=r["data"].get("namerxn"),
        date=r["data"]["date"],
        rxn_smiles=rxn_smiles,
        reactants=matched_reactants,
        products=matched_products,
        agents=matched_agents,
        temperature=temperature,
    )

    return datum


def collect(raw_data: list[dict]) -> tuple[list[ReactionDatum], dict]:
    """Collect from one pickle file of raw data, return list of ReactionDatum and stats

    Parameters:
    ----------
    raw_data: list[dict], list of raw data dictionaries from one pickle file

    Returns:
    -------
    collected_data: list[ReactionDatum], list of collected ReactionDatums with duplicates
    stats: dict, with counts, max num of reagents, reactants, num missing temperature
    """

    cnt = 0
    badrecord_cnt = 0
    reverse_mismatch_cnt = 0
    processed_rxn_cnt = 0
    other_error_cnt = 0

    max_num_reagents = 0
    max_num_reactants = 0
    missing_temperature_num = 0
    collected_data = []

    for r in raw_data:
        try:
            datum = collect_reaction(r)
            collected_data.append(datum)

            cnt += 1
            max_num_reagents = max(max_num_reagents, len(datum.agents))
            max_num_reactants = max(max_num_reactants, len(datum.reactants))
            if datum.temperature is None:
                missing_temperature_num += 1

        # For now, set everything to debug level
        except SmilesError as e:
            logger.debug(f"SmilesError: {e}")
            badrecord_cnt += 1

        except BadRecord as e:
            logger.debug(f"BadRecord: {e}")
            badrecord_cnt += 1

        except ReverseMismatchError as e:
            # parts of rxn_smiles are not covered by components
            logger.debug(f"ReverseMismatchError: {e} rxn: {r['data'].get('smiles')}")
            logger.debug(f"ReverseMismatch data:{e.data}")
            reverse_mismatch_cnt += 1

        except ProcessingError as e:
            logger.debug(f"ProcessingError: {e}")
            processed_rxn_cnt += 1

        except Exception as e:
            logger.debug("Some other error occurred: %s", str(e), exc_info=True)
            other_error_cnt += 1

    stats = {
        "cnt": cnt,
        "badrecord_cnt": badrecord_cnt,
        "reverse_mismatch_cnt": reverse_mismatch_cnt,
        "processed_rxn_cnt": processed_rxn_cnt,
        "other_error_cnt": other_error_cnt,
        "max_num_reagents": max_num_reagents,
        "max_num_reactants": max_num_reactants,
        "missing_temperature_num": missing_temperature_num,
    }
    logger.info("---Collection Stats---")
    logger.info(json.dumps(stats, indent=2))

    return collected_data, stats


def deduplicate(
    records: list[ReactionDatum], max_num_reactants: int, max_num_agents: int
) -> list[ReactionDatum]:
    """Deduplicate records using reaction conditions: rxn_smiles, temperature, all reactants and amount, all agents and amounts

    Parameters:
    ----------
    records: list[ReactionDatum], results of collect() with duplicates

    Returns:
    -------
    deduped_record: list[ReactionDatum], deduplicated results
    """

    # Construct DataFrame from data_rows
    data_rows = []
    for datum in records:
        row_dict = {
            "rxn_smiles": datum.rxn_smiles,
            "temperature": round(datum.temperature, 2) if datum.temperature else None,
        }

        # reactants with padding
        for i, reactant in enumerate(datum.reactants, start=1):
            row_dict[f"reactant_{i}_smi"] = reactant.smiles
            row_dict[f"reactant_{i}_amt"] = (
                round(reactant.amount, 4) if reactant.amount else None
            )
        for i in range(len(datum.reactants) + 1, max_num_reactants + 1):
            row_dict[f"reactant_{i}_smi"] = None
            row_dict[f"reactant_{i}_amt"] = None

        # agents with padding
        for i, agent in enumerate(datum.agents, start=1):
            row_dict[f"agent_{i}_smi"] = agent.smiles
            row_dict[f"agent_{i}_amt"] = (
                round(agent.amount, 4) if agent.amount else None
            )
        for i in range(len(datum.agents) + 1, max_num_agents + 1):
            row_dict[f"agent_{i}_smi"] = None
            row_dict[f"agent_{i}_amt"] = None

        data_rows.append(row_dict)

    df = pd.DataFrame(data_rows)

    # Deduplicate with DataFrame
    df.drop_duplicates(inplace=True)
    deduped_idx = df.index.values
    deduped_record = [records[i] for i in deduped_idx]

    return deduped_record


def _process_chunk(input_path: str, temp_dedup_dir: str) -> None:
    """Helper function to process a single chunk file, same as collect_and_deduplicate_single besides the saving path"""
    print(f"Processing chunk: {os.path.basename(input_path)}", flush=True)

    with open(input_path, "rb") as f:
        raw_data = pickle.load(f)

    collected_data, collect_stats = collect(raw_data)
    deduped_records = deduplicate(
        collected_data,
        collect_stats["max_num_reactants"],
        collect_stats["max_num_reagents"],
    )

    # Save to temporary deduped file
    output_path = os.path.join(
        temp_dedup_dir, f"temp_deduped_{os.path.basename(input_path)}"
    )
    with open(output_path, "wb") as f:
        pickle.dump(deduped_records, f, pickle.HIGHEST_PROTOCOL)


def collect_and_deduplicate_parallel(config: dict) -> None:
    """Run parallel collection and local deduplication process (within each grouped chunk)"""
    logger.info("--- parallel collection and deduplication ---")

    input_dir = config["data_collection"]["input_dir"]
    temp_dedup_dir = config["data_collection"]["temp_dedup_dir"]
    num_workers = config["chunking"]["num_workers"]

    # Create output directory if it doesn't exist
    if not os.path.exists(temp_dedup_dir):
        os.makedirs(temp_dedup_dir)

    # Get all chunk files
    chunk_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith((".pkl", ".pickle"))
    ]
    print(f"Found {len(chunk_files)} chunks to process")

    # Process chunks in parallel
    with Pool(num_workers) as pool:
        pool.starmap(_process_chunk, [(f, temp_dedup_dir) for f in chunk_files])
