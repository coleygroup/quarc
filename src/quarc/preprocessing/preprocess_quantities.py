"""The condition parsing script, needs to be refactored and cleaned"""

import json

import pandas as pd
from rdkit import RDLogger

from quarc.preprocessing.exceptions import *
from quarc.utils.quantity_utils import (
    get_molecular_weight,
    get_solute_solvent,
    preprocess_reagents,
)
from quarc.utils.smiles_utils import canonicalize_smiles
from quarc.settings import load as load_settings

cfg = load_settings()

RDLogger.DisableLog("rdApp.*")

#### Lazy loading for density data ####
_density_clean = None
_reagent_conv_rules = None


def _get_density_data():
    """Lazy loading of density data. Raises error if unavailable and needed."""
    global _density_clean
    if _density_clean is None:
        if cfg.pistachio_density_path is None:
            raise InvalidDensityError(
                "Density file path not configured. Set pistachio_density_path in your QUARC config."
            )
        try:
            _density_clean = pd.read_csv(cfg.pistachio_density_path, sep="\t")
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise InvalidDensityError(
                f"Could not load density file from {cfg.pistachio_density_path}: {e}. "
                "This file is required for quantity preprocessing."
            )
    return _density_clean


def _get_reagent_conv_rules():
    """Lazy loading of reagent conversion rules."""
    global _reagent_conv_rules
    if _reagent_conv_rules is None:
        try:
            with open(cfg.processed_data_dir / "agent_encoder" / "agent_rules_v1.json", "r") as f:
                _reagent_conv_rules = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise FileNotFoundError(
                f"Could not load reagent conversion rules: {e}. "
                "This file is required for preprocessing."
            )
    return _reagent_conv_rules


def get_density(s):
    """Get density from density_clean.tsv, unit is (g/L)"""
    density_clean = _get_density_data()
    if s not in density_clean["can_smiles"].values:
        raise InvalidDensityError(f"smiles={s} not found in density_clean can_smiles")

    return density_clean[density_clean["can_smiles"] == s]["density"].values[0]


def canonicalize_smiles_reagent_conv_rules(s):
    reagent_conv_rules = _get_reagent_conv_rules()
    r = reagent_conv_rules.get(s, None)
    if r is not None:
        return r
    else:
        return s


def getq(qs, s):
    for i in qs:
        if i["type"] == s:
            return i
    return None


def find_quantity_value(component, quantity_type):
    return next(
        (
            q.get("value")
            for q in component.get("quantities", [])
            if q.get("type") == quantity_type
        ),
        None,
    )


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
                                        res[idx_res]["quantities"][i].get("value") == "NaN"
                                        or res[idx_res]["quantities"][i].get("value") is None
                                        or c_q.get("value") == "NaN"
                                        or c_q.get("value") is None
                                    ):
                                        res[idx_res]["quantities"][i]["value"] = None
                                    else:
                                        res[idx_res]["quantities"][i]["value"] += c_q["value"]
                                    res[idx_res]["quantities"][i]["text"] = "should not read"
                            if not found_same_type:
                                res[idx_res]["quantities"].append(c_q)
                    else:
                        res.append(c)
    return res


def determine_category(component):
    """Determine category based on component smiles. The two conditions are:
    1) presence of '.' in smiles
    2) presence of nested component structure
    """
    has_dot = "." in component.get("smiles", "")
    has_nested = "components" in component

    if has_dot and has_nested:
        return "HasDotAndNested"  # mixture, split by inner quantity
    elif has_dot:
        return "HasDotOnly"  # exceptions
    elif has_nested:
        return "HasNestedOnly"  # acid/bases, insert water and split
    else:
        return "Simple"  # do nothing


def is_valid_quantity(component):
    """Valid quantity is defined as either the outer quantity (same level as smiles) has volume
    mass, or amount, or the inner quantity (nested component) has volume. This is to filter out
    invalid cases such as only molarity is given."""
    outer_quantities = component.get("quantities", [])
    if any(q.get("type") in ["Volume", "Mass", "Amount"] for q in outer_quantities):
        return True

    # check for the volume exception in inner quantities
    inner_components = component.get("components", [{}])[0].get("components", [])
    for comp in inner_components:
        if any(q.get("type") == "Volume" for q in comp.get("quantities", [])):
            return True
    return False


def has_inner_qtype(component, qtype):
    """helper function to check if certain quantity type exists in component, including nested components"""
    if "components" in component:
        for sub_component in component["components"]:
            result = has_inner_qtype(sub_component, qtype)
            if result:
                return True
        return False
    else:
        # Check for standard 'type'
        for q in component.get("quantities", []):
            if q.get("type") == qtype:
                return True

            # Special case: if 'type' is missing but 'N' is in 'text'
            if "type" not in q and "N" in q.get("text", ""):
                if qtype == "Normality":
                    return True
        return False
        # return any(q['type'] == qtype for q in component.get('quantities', []))


### Case A: HasDotandNested: process_mixture and all the helper functions


# HasDotAndNested subcase 1: inner quantity specified as ComponentFraction
def handle_fraction_type(component):
    """Used in process_mixture to handle cases where inner quantity is specified as ComponentFraction"""
    outer_volume = next(
        (q.get("value") for q in component.get("quantities", []) if q.get("type") == "Volume"),
        None,
    )
    if not outer_volume:
        raise InvalidQuantError("Outer volume not found in component", data=component)
    nested_components = component.get("components", [])[0].get("components", [])
    constituents = []

    for comp in nested_components:
        fraction = next(
            (q for q in comp.get("quantities", []) if q.get("type") == "ComponentFraction"), None
        )
        if not fraction:
            raise InvalidQuantError(
                "handle_fraction_type: Component fraction not found", data=component
            )

        num, den = map(float, fraction["text"].split("/"))
        vol = outer_volume * (num / den)
        try:
            comp_smi = comp["smiles"]
        except:
            raise SmilesError(f"No smiles for component {comp.get('name')}", data=component)
        s = canonicalize_smiles(comp_smi)
        mole = vol * get_density(s) / get_molecular_weight(s)

        constituents.append(
            {
                "smiles": s,
                "mole": mole,
                "orig_smiles": canonicalize_smiles(component["smiles"]),
                "role": component.get("role"),
            }
        )

    return constituents


# HasDotAndNested subcase 2.1: inner quantity specified as Strength with Molarity
def handle_molarity(strength_value, outer_volume, solutes, solvents):
    """Strength is given as molarity
    If exact 1 solute and 1 solvent, while inner Molarity is given, then calculate related values.
    NOTE: assume solution density equals pure solvent density (no volume change upon mixing)

    Parameters:
    ----------
    strength_value: float, Molarity value
    outer_volume: solution volume
    solutes, solvents: outputs from get_solute_solvent(mixture_smiles)

    Returns:
    -------
    solute_mole: calculated from outer_volume and molarity (strength value)
    solvent_mole: calculated from solution mass and solvent density
    """
    if not (len(solutes) == 1 and len(solvents) == 1):
        raise InvalidSolventError(
            f"inner molarity but not 1 solute and 1 solvent \n \
                        solutes: {solutes}, solvents: {solvents}"
        )

    solution_density = get_density(list(solvents)[0])  # same as pure solvent density!
    solution_mass = outer_volume * solution_density

    solute_mole = outer_volume * strength_value
    solute_mass = get_molecular_weight(list(solutes)[0]) * solute_mole
    solvent_mass = solution_mass - solute_mass
    solvent_mole = solvent_mass / get_molecular_weight(list(solvents)[0])

    return {
        canonicalize_smiles(list(solutes)[0]): solute_mole,
        canonicalize_smiles(list(solvents)[0]): solvent_mole,
    }


CONVENTIONAL_PAIRS = {
    "C=O.O": {37: "w/w"},  # formaldehyde
    "O.O[Na]": {50: "w/w"},  # sodium hydroxide
}


# HasDotAndNested subcase 2.2: inner quantity specified as Strength with Percentage
def handle_percentage(
    strength_value,
    outer_volume,
    outer_mass,
    mixture_smiles,
    comp_smiles,
    solutes,
    solvents,
    strength_text,
):
    """Strength is given as percentage
    1. based on rules or conventions to determine percentage type (w/w, w/v, v/v)
    2. calculate moles for solutes and solvents accordingly based on percentage type

    Parameters:
    ----------
    strength_value: float, Molarity value
    outer_volume: solution volume, expect either outer_volume or outer_mass will be given
    outer_mass: solution mass, expect either outer_volume or outer_mass will be given
    mixture_smiles: used for checking conventional pairs such as 37% Aldehyde usually w/w
    solutes, solvents: outputs from get_solute_solvent(mixture_smiles)
    strength_text: used for checking percentage ypes

    Returns:
    -------
    dict: [{solute_smiles: solute_mole}, {solvent_smiles: solvent_mole}]
    solute_mole: calculated from outer_volume and molarity (strength value)
    solvent_mole: calculated from solution mass and solvent density

    smiles added for v/v case to guarantee solvent order is correct
    """

    # Step 1. determining percentage type
    # Priority 1: Check if strength_text already includes w/w, w/v, or v/v
    strength_text = strength_text.lower().strip()
    wws = ["w/w", "wt.", "wt%", "wt %"]
    wvs = ["w/v"]
    vvs = ["v/v", "vol%", "vol %"]
    if any(i in strength_text for i in wws + wvs + vvs):
        pass

    # Priority 2: Check for conventional pairs
    elif CONVENTIONAL_PAIRS.get(mixture_smiles, {}).get(strength_value):
        strength_text = CONVENTIONAL_PAIRS.get(mixture_smiles, {}).get(strength_value)

    # Priority 3: avaliablity of outer_mass and outer_volume
    elif outer_mass and not outer_volume:
        strength_text = "w/w"

    # Priority 4: Determine based on the number of solutes and solvents
    else:
        if len(solutes) == 1 and len(solvents) == 1:
            strength_text = "w/v"
        elif len(solutes) == 0 and len(solvents) == 2:
            strength_text = "v/v"
        else:
            raise InvalidQuantError(
                f"handle_percentage: Cannot determine percentage type. \n \
                strength_text: {strength_text}, mixture_smiles: {mixture_smiles}, \
                solutes: {solutes}, solvents: {solvents}"
            )

    # Step 2: calculate amounts based on determined type
    solute_mass, solvents_mass = None, None
    # w/w
    if any(i in strength_text for i in wws):
        if not (len(solutes) == 1 and len(solvents) == 1):
            raise InvalidSolventError(
                f"w/w percentage but not 1 solute and 1 solvent \n \
                solutes: {solutes}, solvents: {solvents}"
            )

        if not outer_mass:
            raise InvalidQuantError(
                f"w/w percentage but out outer mass \n \
                strength_text: {strength_text}, mixture_smiles: {mixture_smiles}"
            )

        solute_mass = outer_mass * strength_value / 100
        solvents_mass = outer_mass - solute_mass

        solute_mole = solute_mass / get_molecular_weight(list(solutes)[0])
        solvent_mole = solvents_mass / get_molecular_weight(list(solvents)[0])

        return {
            canonicalize_smiles(list(solutes)[0]): solute_mole,
            canonicalize_smiles(list(solvents)[0]): solvent_mole,
        }
    # w/v
    elif any(i in strength_text for i in wvs):
        if not (len(solutes) == 1 and len(solvents) == 1):
            raise InvalidSolventError(
                f"w/v percentage but not 1 solute and 1 solvent \n \
                    solutes: {solutes}, solvents: {solvents}"
            )

        solute_mass = (
            outer_volume * (strength_value / 100) * 1000
        )  # convert L to mL to get grams for mass
        solute_mole = solute_mass / get_molecular_weight(list(solutes)[0])

        solution_density = get_density(
            list(solvents)[0]
        )  # assume solution density equals pure solvent density (g/L)
        solution_mass = outer_volume * solution_density
        solvent_mass = solution_mass - solute_mass
        solvent_mole = solvent_mass / get_molecular_weight(list(solvents)[0])
        return {
            canonicalize_smiles(list(solutes)[0]): solute_mole,
            canonicalize_smiles(list(solvents)[0]): solvent_mole,
        }
    # v/v
    elif any(i in strength_text for i in vvs):
        if not (len(solutes) == 0 and len(solvents) == 2):
            raise InvalidSolventError(
                f"v/v percentage but not 0 solute and 2 solvents \n \
                    solutes: {solutes}, solvents: {solvents}"
            )
        solvent_1_vol = outer_volume * (strength_value / 100)
        solvent_2_vol = outer_volume - solvent_1_vol

        # order of solvents are not guarranteed, use comp['smiles'] to check
        solvent_1_smi = canonicalize_smiles(comp_smiles)
        all_solvents_smi = set(mixture_smiles.split("."))
        if solvent_1_smi not in all_solvents_smi:
            raise InvalidSolventError(
                f"v/v percentage, comp['smiles']={comp_smiles} not in mixture_smiles={mixture_smiles}"
            )
        solvent_2_smi = next(iter(all_solvents_smi - {solvent_1_smi}), None)

        # just call solvent 1 (with strength info) solute
        solute_mole = (
            get_density(solvent_1_smi) * solvent_1_vol / get_molecular_weight(solvent_1_smi)
        )
        solvent_mole = (
            get_density(solvent_2_smi) * solvent_2_vol / get_molecular_weight(solvent_2_smi)
        )

        return {
            canonicalize_smiles(solvent_1_smi): solute_mole,
            canonicalize_smiles(solvent_2_smi): solvent_mole,
        }


## HasDotAndNested subcase 2: inner quantity specified as Strength (both molarity and percentage)
def handle_strength_type(component):
    outer_volume = find_quantity_value(component, "Volume")
    outer_mass = find_quantity_value(component, "Mass") if not outer_volume else None
    mixture_smiles = canonicalize_smiles(component["smiles"])

    if not (outer_volume or outer_mass):
        raise InvalidQuantError(
            "handle_strength_type: Outer volume or mass not found", data=component
        )
    if len(component["smiles"].split(".")) > 2:
        raise InvalidQuantError(
            f"handle_strength_type: More than 2 components in mixture={mixture_smiles}",
            data=component,
        )

    constituents = []
    nested_components = component.get("components", [])[0].get("components", [])

    for comp in nested_components:
        try:
            comp_smi = comp["smiles"]
        except:
            raise SmilesError(f"No smiles for component {comp.get('name')}", data=component)

        strength = next(
            (q for q in comp.get("quantities", []) if q.get("type") == "Strength"), None
        )
        if strength:
            strength_value = strength["value"]
            strength_text = strength["text"]

            solutes, solvents = get_solute_solvent(component["smiles"])

            if "M" in strength_text:
                smi_mole_dict = handle_molarity(strength_value, outer_volume, solutes, solvents)
            elif "%" in strength_text:
                smi_mole_dict = handle_percentage(
                    strength_value,
                    outer_volume,
                    outer_mass,
                    mixture_smiles,
                    comp_smi,
                    solutes,
                    solvents,
                    strength_text,
                )

            for smi, mole in smi_mole_dict.items():
                constituents.append(
                    {
                        "smiles": smi,
                        "mole": mole,
                        "orig_smiles": mixture_smiles,
                        "role": component.get("role"),
                    }
                )
    return constituents


# HasDotAndNested subcase 3: inner quantity specified as Volume (while no outer volume)
def handle_volume_type(component):
    nested_components = component.get("components", [])[0].get("components", [])
    constituents = []
    for comp in nested_components:
        try:
            comp_smi = comp["smiles"]
        except:
            raise SmilesError(f"No smiles for component {comp.get('name')}", data=component)

        comp_vol = None
        comp_mole = None
        for q in comp.get("quantities", []):
            if q.get("type") == "Volume":
                comp_vol = q["value"]
                comp_mole = comp_vol / get_molecular_weight(comp_smi)
        constituents.append(
            {
                "smiles": canonicalize_smiles(comp_smi),
                "mole": comp_mole,
                "orig_smiles": canonicalize_smiles(component["smiles"]),  # mixture_smi
                "role": component.get("role"),
            }
        )
    return constituents


# HasDotAndNested subcase 4: inner quantity is Nomality, not implemented yet
def handle_normality_type(component):
    raise QuantityError("Normality not implemented yet.")


def process_mixuture(component):
    """Process a chemical mixture to determine its constituents. Splitting and calculations depend on the inner
    quantity type. If no valid quantity is given, just return the split smiles with quantity as None.

    Parameters:
    ----------
        component (dict): Pistachio components for each reaction

    Returns:
    -------
        constituents: A list of dictionaries containing SMILES and moles for each constituent. #TODO: add rdkit mol for each constituent
    """
    constituents = []
    s = canonicalize_smiles(component["smiles"])
    # if no valid quantity, just split and leave the quantity as None
    if not is_valid_quantity(component):
        split_smiles = preprocess_reagents([component["smiles"]])
        return [
            {
                "smiles": canonicalize_smiles(smi),
                "mole": None,
                "orig_smiles": s,
                "role": component.get("role"),
            }
            for smi in split_smiles
        ]

    # if has outer molarity and outer volume, compute directly, no need to use inner quantity
    outer_volume = find_quantity_value(component, "Volume")
    outer_molarity = find_quantity_value(component, "Molarity")
    if outer_molarity and outer_volume:
        solutes, solvents = get_solute_solvent(component["smiles"])
        smi_mole_dict = handle_molarity(outer_molarity, outer_volume, solutes, solvents)
        constituents = [
            {"smiles": smi, "mole": mole, "orig_smiles": s, "role": component.get("role")}
            for smi, mole in smi_mole_dict.items()
        ]
        return constituents

    if has_inner_qtype(component, "Volume"):
        constituents.extend(handle_volume_type(component))
    elif has_inner_qtype(component, "ComponentFraction"):
        constituents.extend(handle_fraction_type(component))
    elif has_inner_qtype(component, "Strength"):
        constituents.extend(handle_strength_type(component))
    elif has_inner_qtype(component, "Normality"):
        constituents.extend(handle_normality_type(component))
    else:
        raise InvalidQuantError(
            "process_mixture: Cannot determine inner quantity type.", data=component
        )

    return constituents


### Case B: HasNestedOnly: process_acidsbases and all the helper functions
def process_acidbase(component):
    """For category HasNestedOnly, the following strategy is used:
    1) add water as solvent to the solute (acid/base)
    2) using handle_molarity and handle_percentage to calculate solute and solvent moles
    """
    constituents = []
    s = canonicalize_smiles(component["smiles"])

    # if no valid quantity, just add water and leave the quantity as None
    if not is_valid_quantity(component):
        return [
            {"smiles": s, "mole": None, "orig_smiles": s, "role": component.get("role")},
            {"smiles": "O", "mole": None, "orig_smiles": s, "role": component.get("role")},
        ]

    solutes = {s}
    solvents = {"O"}

    # Extract quantities and validate
    outer_volume = find_quantity_value(component, "Volume")
    outer_molarity = find_quantity_value(component, "Molarity")
    outer_mass = find_quantity_value(component, "Mass") if not outer_volume else None
    if not (outer_volume or outer_mass):
        raise InvalidQuantError("process_acidbase: Outer volume or mass not found", data=component)

    # Extract strength information
    try:
        strength = component.get("components", [])[0].get("quantities", [])[0]
    except IndexError:  # not strength bc "concentrated/saturated"
        raise InvalidQuantError(
            f"Missing strength (conc. or sat) for {component['name']}", data=component
        )

    strength_value = strength["value"]
    strength_text = strength["text"]

    # Handle Molarity
    if ("M" in strength_text) and (not outer_molarity):
        outer_molarity = strength_value  # find inner molarity if outer molarity not found

    if outer_volume and outer_molarity:
        smi_mole_dict = handle_molarity(outer_molarity, outer_volume, solutes, solvents)
        # original smiles is the solutes only for matching
        constituents = [
            {"smiles": smi, "mole": mole, "orig_smiles": s, "role": component.get("role")}
            for smi, mole in smi_mole_dict.items()
        ]
        return constituents

    # Handle Percentage
    if "%" in strength_text:
        mixture_smiles = canonicalize_smiles(component["smiles"] + ".O")
        smi_mole_dict = handle_percentage(
            strength_value,
            outer_volume,
            outer_mass,
            mixture_smiles,
            s,
            solutes,
            solvents,
            strength_text,
        )
        constituents = [
            {"smiles": smi, "mole": mole, "orig_smiles": s, "role": component.get("role")}
            for smi, mole in smi_mole_dict.items()
        ]
        return constituents

    # Handle Nomality (TBD)
    elif "N" in strength_text:
        pass

    return constituents


### Case C: HasDotOnly: This is exception class
def is_valid_quantity_for_exceptions(component):
    """A more strict version of checking validity of quanity by considering concentration.
    Only applies to HasDotOnly case when no nested component.
    1. If mass or amount -> Valid
    2. If solution, with volume and molarity -> Valid
    3. If one liquid or hydrate with volume only -> Valid

    """
    q_types = {q.get("type") for q in component.get("quantities", [])}

    # case 1: mass or amount present
    if "Amount" in q_types or "Mass" in q_types:
        return True

    # case 2: solution with volume and molarity
    if "Volume" in q_types:
        if "Molarity" in q_types:
            return True

        # case 3: if only volume, has to be one liquid or hydrate
        s = canonicalize_smiles(component["smiles"])
        solutes, solvents = get_solute_solvent(s)
        if len(solvents) == 1 and len(solutes) == 0:
            return True

        # Edge case: check for hydrate in the name
        if "name" in component and "hydrate" in component["name"].lower():
            return True
        return False
    return False


def handle_dup_smiles(smi):
    """
    Handles specific SMILES strings that are known to have unnecessary duplicates.
    """
    DUP_SMILES = {"O.O": "O", "ClCCl.ClCCl.ClCCl": "ClCCl"}

    return DUP_SMILES.get(smi, smi)


# Case 4:
def process_exceptions(component):
    """
    Treat the HasDotOnly category as exceptions because no details for splitting. This category
    contains things like ions, metal complexes (expected to be standardized LATER), and a few halides.
    The strategy is to not split by dot unless neccessay. A more strict valid quantity checking function
    is_valid_quantity_for_exceptions is used to hanlde specials cases.

    Case 1: is mass or amount present -> extract then and convert if needed
    Case 2: both volume and molarity present -> calculate mole for solutes (and solvent if exists)
    Case 3: only volume present -> calculate mole for the liquid or hydrate (special case NN.O)
    """
    constituents = []
    s = handle_dup_smiles(component["smiles"])  # O.O and ClCCl.ClCCl.ClCCl
    s = canonicalize_smiles(s)
    if not s:
        raise SmilesError(f"No smiles for component {component.get('name')}", data=component)

    # if no valid quantity (special rules), leave the quantity as None
    if not is_valid_quantity_for_exceptions(component):
        return [{"smiles": s, "mole": None, "orig_smiles": s, "role": component.get("role")}]

    # valid quantity: mole or mass
    amount = find_quantity_value(component, "Amount")
    if amount:
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    mass = find_quantity_value(component, "Mass")
    if mass:
        amount = mass / get_molecular_weight(s)
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    # valid quantity: vol + molarity
    volume = find_quantity_value(component, "Volume")
    molarity = find_quantity_value(component, "Molarity")
    if volume and molarity:
        solutes, solvents = get_solute_solvent(s)

        # if solutes only
        if len(solutes) > 0 and len(solvents) == 0:
            return [
                {
                    "smiles": s,
                    "mole": molarity * volume,
                    "orig_smiles": s,
                    "role": component.get("role"),
                }
            ]

        # if typical 1 solute + 1 solvent
        smi_mole_dict = handle_molarity(molarity, volume, solutes, solvents)
        constituents = [
            {"smiles": smi, "mole": mole, "orig_smiles": s, "role": component.get("role")}
            for smi, mole in smi_mole_dict.items()
        ]

    # valid quantity: vol only for liquid or hydrate (edge case)
    if volume and (not molarity):
        mass = get_density(s) * volume
        amount = mass / get_molecular_weight(s)
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    return constituents


def process_simple(component):
    """
    Treat the Simple category. For quantity checking, this would continue to use the strict version.
    The key difference between process_simple and process_exception is that in case 2, volume and
    molarity is used to compute for solutes only (no involvement of solvent at all).

    Case 1: is mass or amount present -> extract then and convert if needed
    Case 2: both volume and molarity present -> calculate mole for solutes
    Case 3: only volume present -> calculate mole for the liquid or hydrate (special case NN.O)
    """
    constituents = []
    s = canonicalize_smiles(component.get("smiles"))
    if not s:
        raise InvalidQuantError("process_simple: component no smiles", data=component)

    # if no valid quantity (special rules), leave the quantity as None
    if not is_valid_quantity_for_exceptions(component):
        return [{"smiles": s, "mole": None, "orig_smiles": s, "role": component.get("role")}]

    # valid quantity: mole or mass
    amount = find_quantity_value(component, "Amount")
    if amount:
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    mass = find_quantity_value(component, "Mass")
    if mass:
        amount = mass / get_molecular_weight(s)
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    # valid quantity: vol + molarity,
    volume = find_quantity_value(component, "Volume")
    molarity = find_quantity_value(component, "Molarity")
    if volume and molarity:
        return [
            {
                "smiles": s,
                "mole": molarity * volume,
                "orig_smiles": s,
                "role": component.get("role"),
            }
        ]

    # valid quantity: vol only for liquid or hydrate (edge case)
    if volume and (not molarity):
        mass = get_density(s) * volume
        amount = mass / get_molecular_weight(s)
        return [{"smiles": s, "mole": amount, "orig_smiles": s, "role": component.get("role")}]

    return constituents
