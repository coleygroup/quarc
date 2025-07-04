from __future__ import print_function

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Common solvents (Remove Pistachio's density data)
density_clean = None

solvents_verified = [
    "pentane",
    "hexane",
    "heptane",
    "octane",
    "diethyl ether",
    "triethylamine",
    "decane",
    "methyl tert-butyl ether",
    "cyclopentane",
    "cyclohexane",
    "acetone",
    "acetonitrile",
    "2-propanol",
    "ethanol",
    "tert-butanol",
    "methanol",
    "isobutanol",
    "1-propanol",
    "2-butanone",
    "2-pentanone",
    "2-butanol",
    "1-butanol",
    "2-pentanol",
    "cyclohexene",
    "1-pentanol",
    "3-pentanone",
    "1-heptanol",
    "1-octanol",
    "1-nonanol",
    "1-decanol",
    "m-xylene",
    "p-xylene",
    "cumene",
    "ethylbenzene",
    "toluene",
    "1,2-dimethoxyethane",
    "benzene",
    "benzonitrile",
    "o-xylene",
    "THF",
    "hexamethylphosphorus triamide",
    "ethyl acetate",
    "ethyl formate",
    "methyl acetate",
    "diglyme",
    "cyclohexanone",
    "DMF",
    "cyclohexanol",
    "diethyl carbonate",
    "methyl formate",
    "pyridine",
    "anisole",
    "water",
    "aniline",
    "1,4-dioxane",
    "hexamethylphosphoramide",
    "N-methyl-2-pyrrolidone",
    "naphthalene",
    "benzaldehyde",
    "benzyl alcohol",
    "acetic acid",
    "dimethyl carbonate",
    "DMSO",
    "2-phenoxyethanol",
    "chlorobenzene",
    "heavy water",
    "ethylene glycol",
    "diethylene glycol",
    "1,2-dichloroethane",
    "formic acid",
    "1,2-dichloroethane",
    "glycerol",
    "carbon disulfide",
    "DCM",
    "nitromethane",
    "chloroform",
    "TFA",
    "carbon tetrachloride",
    "hydrazine hydrate",
    "isoporpyl alcohol"
    # 16 newly added solv with densities
    "deuterochloroform",
    "tert-Amyl alcohol",
    "2-ethoxyethanol",
    "propionitrile",
    "mesitylene",
    "Isohexane",
    "dimethylether",
    "trifluoromethylbenzene",
    "Dowtherm",
    "butyronitrile",
    "tert-butyl acetate",
    "propylene glycol",
    "1,1,2,2-tetrachloroethane",
    "1,4-dichlorobenzene",
    "triethylene glycol",
    "p-cymene",
]
common_solvents_list = density_clean[density_clean["name"].isin(solvents_verified)][
    "can_smiles"
].tolist()
COMMON_SOLVENTS_CANONICAL = set(common_solvents_list)


SMILES2CAS = {}
SMILES2CAS_fn = "smiles2cas.json"
if os.path.exists(SMILES2CAS_fn) and os.path.isfile(SMILES2CAS_fn):
    with open(SMILES2CAS_fn, "r") as f:
        SMILES2CAS = json.load(f)


def get_morgan_fp(s, fp_radius, fp_length):
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s, sanitize=True), fp_radius, nBits=fp_length
        )
    )


def num_mol(smiles):
    smiles_splitted = smiles.split(".")
    return len(smiles_splitted)


def is_atom_map_mol(mol):
    res = False
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            res = True
            break
    return res


def is_atom_map_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return is_atom_map_mol(mol)


def is_atom_map_rxn_smiles(s):
    s_list = s.split(" ")
    r = AllChem.ReactionFromSmarts(str(s_list[0]))
    products = r.GetProducts()
    res = False
    for p in products:
        if is_atom_map_mol(p):
            res = True
            break
    return res


def remove_unmapped_in_mapped_rxn(reaction_smiles):
    """Remove unmapped reactants in atom-mapped smiles to the context section"""

    def filter_atom_mapped_by_dot(smiles):
        components = smiles.split(".")
        mapped_components = [comp for comp in components if ":" in comp]
        unmapped_components = [comp for comp in components if ":" not in comp]
        return ".".join(mapped_components), unmapped_components

    reactants, reagents, products = reaction_smiles.split(">")

    reagent_set = set(reagents.split("."))
    filtered_reactants, unmapped_reactants = filter_atom_mapped_by_dot(reactants)
    filtered_products, unmapped_products = filter_atom_mapped_by_dot(products)

    unique_extra_unmapped = set(unmapped_reactants) - reagent_set
    if unique_extra_unmapped:
        reagents = ".".join(filter(None, [reagents, ".".join(unique_extra_unmapped)]))
    return f"{filtered_reactants}>{reagents}>{filtered_products}"


def get_atom_map_asdict(mol, reverse=False):
    """Return {atom_idx:map_num}. If reverse is true, return {map_num:atom_idx, None:[all_other]}"""
    res = {}
    if reverse:
        res = {None: []}
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                res[None].append(atom.GetIdx())
            else:
                res[map_num] = atom.GetIdx()
    else:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            res[atom.GetIdx()] = None if map_num == 0 else map_num
    return res


def canonicalize_smiles_rdkit(s):
    try:
        s_can = Chem.MolToSmiles(Chem.MolFromSmiles(s, sanitize=False), canonical=True)
    except:
        sys.stderr.write("canonicalize_smiles_rdkit(): fail s=" + s + "\n")
        s_can = None
    return s_can


def canonicalize_smiles(s, is_sanitize=True):
    """Remove atom mapping, canonicalize smiles"""
    if s is None:
        return None
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        if is_sanitize:
            Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def remove_stereo(smiles) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def canonicalize_smiles_list(s_list):
    for i in range(len(s_list)):
        s_list[i] = canonicalize_smiles(s_list[i])
    return s_list


def clear_atom_map(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    return mol


def remove_none(l):
    res = []
    for i in l:
        if i is not None:
            res.append(i)
    return res


def combine_smiles(smiles_list, idx):
    """Given a list of smiles, merge the smiles indicated by list idx"""
    if len(idx) <= 1:
        raise Exception("combine_smiles(): len(idx)=" + str(len(idx)))
    idx = list(sorted(idx))
    for i in range(1, len(idx)):
        smiles_list[idx[0]] += "." + smiles_list[idx[i]]
        smiles_list[idx[i]] = None
    return smiles_list


def prep_rxn_smi_input(rxn_smi):
    """remove the |f| part and agents from rxn_smiles"""
    clean_smi = rxn_smi.split(" |")[0]
    reactants, _, products = clean_smi.split(">")
    return f"{reactants}>>{products}"


def split_reaction_smiles(s):
    """split reaction smiles into reactants, reagents and products"""
    # remove extension f:
    s_list = s.split(" ")
    r = str(s_list[0]).split(">")
    reactants = r[0].split(".")
    reagents = r[1].split(".")
    products = r[2].split(".")
    if len(reagents) == 1 and reagents[0] == "":
        reagents = []
    reactants_smiles = [canonicalize_smiles(i) for i in reactants]
    reagents_smiles = [canonicalize_smiles(i) for i in reagents]
    products_smiles = [canonicalize_smiles(i) for i in products]
    # Note: (*)_smiles will change sizes, but (*) will not
    if len(s_list) > 1:
        # has |f:|
        s = s_list[1].strip()
        if s[0:3] == "|f:" and s[-1] == "|":
            s = s[3:-1]
            idx_mol = s.split(",")
            for idx_group in idx_mol:
                idx = idx_group.split(".")
                idx = [int(i) for i in idx]
                idx = list(sorted(idx))
                if idx[0] < len(reactants):
                    if idx[-1] >= len(reactants):
                        raise Exception(
                            "split_reaction_smiles():"
                            + " len(reactants)="
                            + str(len(reactants))
                            + " idx="
                            + str(idx)
                        )
                    reactants_smiles = combine_smiles(reactants_smiles, idx)
                elif idx[0] < len(reactants) + len(reagents):
                    if idx[-1] >= len(reactants) + len(reagents):
                        raise Exception(
                            "split_reaction_smiles():"
                            + " len(reactants)="
                            + str(len(reactants))
                            + " len(reagents)="
                            + str(len(reagents))
                            + " idx="
                            + str(idx)
                        )
                    idx = [i - len(reactants) for i in idx]
                    reagents_smiles = combine_smiles(reagents_smiles, idx)
                else:
                    if idx[-1] >= len(reactants) + len(reagents) + len(products):
                        raise Exception(
                            "split_reaction_smiles():"
                            + " len(reactants)="
                            + str(len(reactants))
                            + " len(reagents)="
                            + str(len(reagents))
                            + " len(products)="
                            + str(len(products))
                            + " idx="
                            + str(idx)
                        )
                    idx = [i - (len(reactants) + len(reagents)) for i in idx]
                    products_smiles = combine_smiles(products_smiles, idx)
    reactants_smiles = remove_none(reactants_smiles)
    reagents_smiles = remove_none(reagents_smiles)
    products_smiles = remove_none(products_smiles)
    canonicalize_smiles_list(reactants_smiles)
    canonicalize_smiles_list(reagents_smiles)
    canonicalize_smiles_list(products_smiles)
    return reactants_smiles, reagents_smiles, products_smiles
