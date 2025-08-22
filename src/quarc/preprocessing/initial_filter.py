import json
import pickle
import os
from typing import Callable

from loguru import logger
from rdkit import Chem, RDLogger
from tqdm import tqdm

from quarc.data.datapoints import ReactionDatum
from quarc.models.modules.agent_encoder import AgentEncoder
from quarc.models.modules.agent_standardizer import AgentStandardizer
from quarc.utils.smiles_utils import get_common_solvents_canonical

RDLogger.DisableLog("rdApp.*")
### Potential filters (True -> keep, False -> remove) ###

COMMON_SOLVENTS_CANONICAL = get_common_solvents_canonical()


# Length filters
def filter_by_agent_length(rxn: ReactionDatum, min_len: int = 0, max_len: int = 5) -> bool:
    """skip if over max_len agents or under min_len agents"""
    return min_len <= len(rxn.agents) <= max_len


def filter_by_reactant_length(rxn: ReactionDatum, min_len: int = 1, max_len: int = 5) -> bool:
    """skip if over max_len reactants or under min_len reactants"""
    return min_len <= len(rxn.reactants) <= max_len


def filter_by_product_length(rxn: ReactionDatum, min_len: int = 1, max_len: int = 1) -> bool:
    """skip if over max_len products or under min_len products. Default is 1 product so min=max=1"""
    return min_len <= len(rxn.products) <= max_len


# Molecule filters
def filter_by_reactant_num_atoms(rxn: ReactionDatum, max_num_atoms: int = 50) -> bool:
    """skip if any reactant has more than max_num_atoms"""
    reactant_mols = [Chem.MolFromSmiles(r.smiles) for r in rxn.reactants]
    if any(mol is None for mol in reactant_mols):
        return False
    return all(mol.GetNumAtoms() <= max_num_atoms for mol in reactant_mols)


def filter_by_product_num_atoms(rxn: ReactionDatum, max_num_atoms: int = 50) -> bool:
    """skip if any product has more than max_num_atoms"""
    product_mols = [Chem.MolFromSmiles(p.smiles) for p in rxn.products]
    if any(mol is None for mol in product_mols):
        return False
    return all(mol.GetNumAtoms() <= max_num_atoms for mol in product_mols)


# Reaction quality
def filter_by_mol_parsability(rxn: ReactionDatum) -> bool:  # FIXME: implicitly checked before
    """skip if any reactant or product is not parsable by RDKit, mainly for FP + MLP"""
    for r in rxn.reactants:
        if Chem.MolFromSmiles(r.smiles) is None:
            return False

    for p in rxn.products:
        if Chem.MolFromSmiles(p.smiles) is None:
            return False

    return True


def filter_by_rxn_smiles_parsability(rxn: ReactionDatum) -> bool:
    """skip if full reaction SMILES cannot be parsed by RDKit, mainly for GNN encoding"""
    clean_smi = rxn.rxn_smiles.split(" |")[0]
    rct_smi, _, pdt_smi = clean_smi.split(">")
    rct_mol = Chem.MolFromSmiles(rct_smi)
    pdt_mol = Chem.MolFromSmiles(pdt_smi)
    return rct_mol is not None and pdt_mol is not None


def filter_by_temp_and_agent(rxn: ReactionDatum) -> bool:
    """skip if no temperature and no agent (means not enough information)"""
    return False if (rxn.temperature is None and len(rxn.agents) == 0) else True


# Rare agents (agent encodability)
def filter_by_agentencoder(
    rxn: ReactionDatum, agent_standardizer: AgentStandardizer, agent_encoder: AgentEncoder
) -> bool:
    """skip unencoable agents using agent encoder (including the "other" category)"""
    try:
        agents_smiles = [agent.smiles for agent in rxn.agents]
        standardized_agents_smiles = agent_standardizer.standardize(agents_smiles)
        agent_encoder.encode(standardized_agents_smiles)
    except Exception:
        return False
    return True


# Solvent filters
def filter_by_solvent_existence(rxn: ReactionDatum) -> bool:
    """skip if none of the reactants or agents are in the common solvent list"""
    reactant_smiles = {r.smiles for r in rxn.reactants}
    agent_smiles = {a.smiles for a in rxn.agents}

    return bool(
        reactant_smiles.intersection(COMMON_SOLVENTS_CANONICAL)
        or agent_smiles.intersection(COMMON_SOLVENTS_CANONICAL)
    )


# Reaction class
def filter_by_recognized_reaction(rxn: ReactionDatum) -> bool:
    """Skip if reaction class is unrecognized (marked as '0.0')"""
    return rxn.rxn_class != "0.0"


# Main filter functions
def apply_filter_group(
    data: list[ReactionDatum], filters: list[tuple[Callable, str]], group_name: str
) -> list[ReactionDatum]:
    """Apply filter and log.
    Filter function returns: True -> keep, False -> remove
    """
    initial_count = len(data)

    filtered_data = []
    for datum in tqdm(data, desc=f"{group_name} filtering"):
        for filter_func, _filter_name in filters:
            if not filter_func(datum):
                break
        else:
            filtered_data.append(datum)

    final_count = len(filtered_data)

    stats = {
        "len_before": initial_count,
        "len_after": final_count,
        "removed_ratio": round((initial_count - final_count) / initial_count, 3),
    }
    logger.info(f"{group_name} filtering results:\n" + json.dumps(stats, indent=2))
    data.clear()
    return filtered_data, stats


def run_initial_filters(config):
    logger.info("---Running initial filters---")
    # Load config parameters
    input_path = config["initial_filter"]["input_path"]
    output_path = config["initial_filter"]["output_path"]

    # Filter parameters
    length_params = config["initial_filter"]["length_filters"]
    product_length = length_params["product"]
    reactant_length = length_params["reactant"]
    agent_length = length_params["agent"]

    atom_params = config["initial_filter"]["atom_filters"]
    max_reactant_atoms = atom_params.get("max_reactant_atoms", 50)
    max_product_atoms = atom_params.get("max_product_atoms", 50)

    agent_standardizer = AgentStandardizer(
        conv_rules=config["generate_agent_class"]["conv_rules_path"],
        other_dict=config["generate_agent_class"]["output_other_dict_path"],
    )
    agent_encoder = AgentEncoder(class_path=config["generate_agent_class"]["output_encoder_path"])

    # Load data
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    initial_count = len(data)

    filter_metadata = {}

    # group 1: length filters
    length_filters = [
        (
            lambda rxn: filter_by_product_length(
                rxn, min_len=product_length["min"], max_len=product_length["max"]
            ),
            "product_length",
        ),
        (
            lambda rxn: filter_by_reactant_length(
                rxn, min_len=reactant_length["min"], max_len=reactant_length["max"]
            ),
            "reactant_length",
        ),
        (
            lambda rxn: filter_by_agent_length(
                rxn, min_len=agent_length["min"], max_len=agent_length["max"]
            ),
            "agent_length",
        ),
    ]
    data, stats = apply_filter_group(data, length_filters, "length")
    filter_metadata["length_filters"] = {
        "filters_applied": [name for _, name in length_filters],
        "stats": stats,
    }

    # group 2: molecule filters
    molecule_filters = [
        (
            lambda rxn: filter_by_reactant_num_atoms(rxn, max_num_atoms=max_reactant_atoms),
            "reactant_num_atoms",
        ),
        (
            lambda rxn: filter_by_product_num_atoms(rxn, max_num_atoms=max_product_atoms),
            "product_num_atoms",
        ),
    ]
    data, stats = apply_filter_group(data, molecule_filters, "molecule")
    filter_metadata["molecule_filters"] = {
        "filters_applied": [name for _, name in molecule_filters],
        "stats": stats,
    }

    # group 3: reaction quality filters
    reaction_quality_filters = [
        (filter_by_mol_parsability, "R_and_P_parsability"),
        (filter_by_temp_and_agent, "no_temp_and_no_agent"),
        (filter_by_rxn_smiles_parsability, "rxn_smiles_parsability"),
    ]
    data, stats = apply_filter_group(data, reaction_quality_filters, "reaction_quality")
    filter_metadata["reaction_quality_filters"] = {
        "filters_applied": [name for _, name in reaction_quality_filters],
        "stats": stats,
    }

    # * group 4: agent encodability
    agent_encoder_filters = [
        (
            lambda rxn: filter_by_agentencoder(rxn, agent_standardizer, agent_encoder),
            "agent_encodability",
        ),
    ]
    data, stats = apply_filter_group(data, agent_encoder_filters, "agent_encodability")
    filter_metadata["agent_encoder_filters"] = {
        "filters_applied": [name for _, name in agent_encoder_filters],
        "stats": stats,
    }

    # * group 5: solvent existence
    solvent_filters = [
        (filter_by_solvent_existence, "solvent_existence"),
    ]
    data, stats = apply_filter_group(data, solvent_filters, "solvent_existence")
    filter_metadata["solvent_filters"] = {
        "filters_applied": [name for _, name in solvent_filters],
        "stats": stats,
    }

    # * group 6: reaction class
    reaction_class_filters = [
        (filter_by_recognized_reaction, "reaction_class"),
    ]
    data, stats = apply_filter_group(data, reaction_class_filters, "reaction_class")
    filter_metadata["reaction_class_filters"] = {
        "filters_applied": [name for _, name in reaction_class_filters],
        "stats": stats,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    parent_dir = "/".join(output_path.split("/")[:-1])
    with open(os.path.join(parent_dir, "initial_filter_metadata.json"), "w") as f:
        json.dump(filter_metadata, f, indent=2)

    final_count = len(data)
    logger.info(
        f"Overall filtering results:\n"
        f"\tbefore initial filtering: {initial_count}\n"
        f"\tafter initial filtering: {final_count}\n"
        f"\t% removed: {(initial_count - final_count)/initial_count:.1%}"
    )
