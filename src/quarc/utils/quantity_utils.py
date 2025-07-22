import json
import math
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from quarc.preprocessing.exceptions import *
from quarc.quarc.utils.smiles_utils import COMMON_SOLVENTS_CANONICAL, canonicalize_smiles
from quarc.config import PROCESSED_DATA_DIR

RDLogger.logger().setLevel(RDLogger.ERROR)

reagent_conv_rules = None
with open(PROCESSED_DATA_DIR / "agent_encoder/agent_rules_v1.json", "r") as f:
    reagent_conv_rules = json.load(f)


def canonicalize_smiles_reagent_conv_rules(s):
    r = reagent_conv_rules.get(s, None)
    if r is not None:
        return r
    else:
        return s


INVALID_SMILES = [
    "null",
    "S=[NH4+]",
    "C[ClH]",
    "[OH2-]",
    "[ClH-]",
    "[P+3]",
    "[B]",
]


def is_smiles_valid(s):
    return s not in INVALID_SMILES


# re-canonicalized
SMILES2CHARGE = {
    "Br[Br-]Br": -1,
    "[BH]1234[H][BH]561[BH]173[BH]34([H]2)[BH]247[BH]761[BH]165[H][BH]586[BH]32([BH]4715)[H]8": 0,
    "[BH3]S(C)C": 0,
    "[BH3-][S+](C)C": 0,
    "[BH3]O1CCCC1": 0,
    "[BH3]N1CCCCC1C": 0,
}


def get_smiles_charge(s):
    q = 0
    for i in s.split("."):
        q += get_smiles_charge_single_mol(i)
    return q


def get_smiles_charge_single_mol(s):
    if not is_smiles_valid(s):
        raise ProcessingError("get_smiles_charge(): s=" + s)
    q = SMILES2CHARGE.get(s, None)
    if q is None:
        try:
            mol = Chem.MolFromSmiles(canonicalize_smiles_reagent_conv_rules(s), sanitize=False)
        except:
            raise ProcessingError("get_smiles_charge(): MolFromSmiles() fails, s=" + s)
        if mol is None:
            raise ProcessingError("get_smiles_charge(): MolFromSmiles() fails, s=" + s)
        q = Chem.rdmolops.GetFormalCharge(mol)
    return q


# re-canonicalized
SMILES2MW = {
    "Br[Br-]Br": 239.71,
    "[H]1[BH]234[H][BH]256[BH]278[H][BH]29%10[H][BH]92%11[BH]139[BH]451[BH]923[BH]%10%117[BH]6813": 112.142,
    "[BH3]S(C)C": 75.97,
    "[BH3-][S+](C)C": 75.97,
    "[BH3]O1CCCC1": 85.94,
    "[BH3]N1CCCCC1C": 113.009,
    "[BH4][Na]": 37.84,
    "[Li][BH4]": 21.8,
    "[BH4][K]": 53.94,
    "Cl1[Rh]Cl[Rh]1": 276.71,
    "CC1=CC(C)=O[Fe]23(O1)(OC(C)=CC(C)=O2)OC(C)=CC(C)=O3": 356.19,
    "[Li]N([Si](C)(C)C)[Si](C)(C)C": 167.4,
    "C[Si](C)(C)N([Na])[Si](C)(C)C": 183.37,
}


def get_molecular_weight(smiles):
    res = 0
    for s in smiles.split("."):
        mw = SMILES2MW.get(s, None)
        if mw is None:
            try:
                mol = Chem.MolFromSmiles(s, sanitize=True)
                mw = Descriptors.MolWt(mol)
            except:
                pass
        if mw is None:
            raise InvalidQuantError(f"get_molecular_weight(): no MW data, smiles={smiles}, s={s}")
            # sys.stderr.write('get_molecular_weight(): no MW data, smiles='+smiles+', s='+s+'\n')
            return None
        res += mw
    return res


def __test_get_molecular_weight():
    smiles_kcl = "[K+].[Cl-]"
    kcl = get_molecular_weight(smiles_kcl)
    assert abs(kcl - 74.551) < 0.01


SOLUTES = [
    "[Li]CCCC",
    "[Li]C(C)(C)C",
    "[Li]C(C)CC",
    "[Li]C",
    "CC(C)[Mg]Cl",
    "CC(C)[Mg]Br",
    "CC[Zn]CC",
    "B",
    "CC(C)C[AlH]CC(C)C",
    "C[Si](C)(C)C=[N+]=[N-]",
    "[AlH4-].[Li+]",
    "B1C2CCCC1CCC2",
    "C[Si](C)(C)[N-][Si](C)(C)C.[Na+]",
    "Cl",
    "Br",
    "[K+].[OH-]",
    "[Na+].[OH-]",
    "N",
    "O=C([O-])O.[Na+]",
    "O=C([O-])[O-].[K+].[K+]",
    "O=C([O-])[O-].[Na+].[Na+]",
]

# re-canonicalized
CHEMICAL_PHASE = {
    "[Li]CCCC": "l",
    "[Li]C(C)(C)C": "l",
    "CC(C)[Mg]Cl": "l",
    "CC[Zn]CC": "l",
    "B": "g",
    "CC(C)C[AlH]CC(C)C": "l",
    "C[Si](C)(C)C=[N+]=[N-]": "l",
    "[AlH4-].[Li+]": "s",
    "[BH4-].[Li+]": "s",
    "[BH3]O1CCCC1": "s",
    "B1C2CCCC1CCC2": "s",
    "[BH3-]C#N.[Na+]": "s",
    "Cl[Pd](Cl)([PH](c1ccccc1)(c1ccccc1)c1ccccc1)[PH](c1ccccc1)(c1ccccc1)c1ccccc1": "s",
    "C[Si](C)(C)[N-][Si](C)(C)C.[Na+]": "s",
    "C[Si](C)(C)[N-][Si](C)(C)C.[Li+]": "s",
    "[SiH3]c1ccccc1": "l",
    "Cl[Ti](Cl)(Cl)Cl": "s",
    "CCC(C)[BH-](C(C)CC)C(C)CC.[Li+]": "s",
    "[BH3]S(C)C": "l",
    "[BH3-][S+](C)C": "l",
    "[NH4+].[NH4+].[S-2]": "s",
    "CCS(N)(F)(F)(F)CC": "s",
    "Cl[Fe](Cl)Cl.O.O.O.O.O.O": "s",
    "[Ag]O[Ag]": "s",
    "N#C[Zn]C#N": "s",
    "COc1ccc(CNc2cc(Oc3ccc(NC(=O)NC(=O)Cc4ccc(F)cc4)cc3F)ncn2)cc1": "l",
    "CC[BH-](CC)CC.[Li+]": "l",
    "[Li][BH](CC)(CC)CC": "l",
    "B.CSC": "l",
    "Cl[SiH](Cl)Cl": "l",
    "CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]": "s",
    "CC(C)C[Al]CC(C)C": "l",
    "O=[Mo](=O)([O-])[O-].[NH4+].[NH4+]": "s",
    "CC(C)(C)O[Al-](OC(C)(C)C)OC(C)(C)C.[Li+]": "s",
    "CCN(CC)CC.Cl": "s",
    "CC[NH+](CC)CC.[Cl-]": "s",
    "O=S(=O)([O-])OO.[K+]": "s",
    "O=S(=O)([O-])O[O-].[K+].[K+]": "s",
    "[Cl-].[NH3+]O": "s",
    "CCN=C=N": "l",
    "[Cu]O[Cu]": "s",
    "O[Pd]O": "s",
    "[Na+].[Na+].[S-2]": "s",
    "CN(C)C(n1n[n+]([O-])c2ncccc21)=[N+](C)C.F[P-](F)(F)(F)(F)F": "s",
    "CC(C)c1cc(C(C)C)c(-c2cccc(P(C3CCCCC3)C3CCCCC3)c2)c(C(C)C)c1": "s",
    "CC(=O)O[BH-](OC(C)=O)OC(C)=O.C[N+](C)(C)C": "s",
    "O=S([O-])OS(=O)[O-].[Na+].[Na+]": "s",
    "COCCO[AlH2-]OCCOC.[Na+]": "s",
    "c1ccc(P(c2ccccc2)C2CCCC2)cc1": "s",
    "C1=NCCCN(C2CCCCCCCCCC2)CCCCC1": "s",
    "Cl[Sn]Cl.O.O": "s",
    "[CH]1[CH][CH][C](P(c2ccccc2)c2ccccc2)[CH]1": "s",
    "CC1(C)OBOC1(C)C": "l",
    "CC(C)[SiH](C(C)C)C(C)C": "l",
    "Cl[Mg]c1ccccc1": "s",
    "O=[Os]": "s",
    "CCC(C)[BH-](C(C)CC)C(C)CC.[K+]": "s",
    "C1CC[NH2+]CC1.CC(=O)[O-]": "s",
    "CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F": "s",
    "CCCC[B+]CCCC.O=S(=O)([O-])C(F)(F)F": "s",
    "CCCP1(=O)OP(=O)(CCC)OP(=O)(CCC)O1": "s",
    "CB1OC(c2ccccc2)(c2ccccc2)[C@@H]2CCCN12": "s",
    "CCC(C)(C)[O-].[Na+]": "s",
    "CCCCP(=CC#N)(CCCC)CCCC": "s",
    "CC[O-].[Na+]": "s",
    "CCCC[N+](CCCC)(CCCC)CCCC.[OH-]": "s",
    "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21": "s",
    "CCCC[Sn](=O)CCCC": "s",
    "CC[SiH](CC)CC": "l",
    "[N-]=[N+]=NP(=O)(c1ccccc1)c1ccccc1": "s",
    "CC(=O)OC=O": "l",
    "CNC1CCCCC1NC": "s",
    "CC1(C)OO1": "g",
    "OCCC1CCCO1": "l",
    "CCO[Ti](OCC)(OCC)OCC": "l",
    "COCCN(CCOC)S(F)(F)F": "l",
    "CC(C)CN1CCN2CCN(CC(C)C)P1N(CC(C)C)CC2": "s",
    "C1CCC(CNCC2CCCCC2)CC1": "l",
    "COB1OC(C)(C)C(C)(C)O1": "l",
    "CN(C)C(=NC(C)(C)C)N(C)C": "l",
    "CC(C)O[Ti](OC(C)C)(OC(C)C)OC(C)C": "l",
    "Cl[Sn]Cl": "s",
    "CC(C)(C)OC(=N)C(Cl)(Cl)Cl": "l",
    "CN(C)[C@@H]1CCCC[C@H]1N": "s",
    "COP(=O)(OC)C(=[N+]=[N-])C(C)=O": "l",
    "CCCP(=O)(O)OP(=O)(O)CCC": "l",
    "NC(C1CCCCC1)C1CCCCC1": "l",
    "CCCC[Sn](CCCC)(CCCC)N=[N+]=[N-]": "l",
    "CN[C@H]1CCCC[C@@H]1NC": "s",
    "CC(=N[Si](C)(C)C)O[Si](C)(C)C": "l",
    "CC(C)(C)OC(=O)N=NC(=O)OC(C)(C)C": "s",
    "O=C(N=NC(=O)N1CCCCC1)N1CCCCC1": "s",
    "CCN=C=NCCCN(C)C": "l",
    "C1CCC2=NCCCN2CC1": "l",
    "[Li]C(C)CC": "l",
    "[Li]C": "l",
    "COC1CCCC1": "l",
    "CC(C)OC(=O)N=NC(=O)OC(C)C": "l",
    "CC[Mg]Br": "l",
    "CCOC(=O)OC(=O)OCC": "l",
    "CN(C)C(=N)N(C)C": "l",
    "CN(C)C(OC(C)(C)C)OC(C)(C)C": "l",
    "CC(C)OC(=O)/N=N/C(=O)OC(C)C": "l",
    "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21": "s",
    "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1": "s",
    "O=[Cu]": "s",
    "CN(C)C(=O)N=NC(=O)N(C)C": "s",
    "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1": "s",
    "Cc1cc(C)c(N2CCN(c3c(C)cc(C)cc3C)C2=[Ru](Cl)(Cl)(=Cc2ccccc2)[PH](C2CCCCC2)(C2CCCCC2)C2CCCCC2)c(C)c1": "s",
    "O=[Pt]=O": "s",
    "O=C(C=Cc1ccccc1)C=Cc1ccccc1": "s",
    "O=C(O[Na])O[Na]": "s",
    "O=C(O[K])O[K]": "s",
    "O=C(O[Na])C(O)C(O)C(=O)O[K]": "s",
    "O=S(O[Na])O[Na]": "s",
    "[Li][AlH4]": "s",
    "[Li][BH4]": "s",
    "[BH4][K]": "s",
    "O=S(O[Na])S(=O)(=O)O[Na]": "s",
    "CC[O+](CC)[B-](F)(F)F": "l",
    "CCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCC(=O)[O-].CCCC[Sn+2]CCCC": "l",
    "O=[N+]([O-])O[Ce](O[N+](=O)[O-])(O[N+](=O)[O-])O[N+](=O)[O-]": "s",
    "O=S(O[Na])S(=O)O[Na]": "s",
    "COCCO[AlH2]([Na])OCCOC": "s",
    "O=S1(=O)O[Cu]O1": "s",
    "O=P(O)(O[K])O[K]": "s",
    "CB1OC(c2ccccc2)(c2ccccc2)C2CCCN12": "s",
    "CC1=CC(C)=O[Fe]23(O1)(OC(C)=CC(C)=O2)OC(C)=CC(C)=O3": "s",
    "O=S1(=O)O[Mg]O1": "s",
    "O=P(O)(O[Na])O[Na]": "s",
    "O=P(O[Na])(O[Na])O[Na]": "s",
    "Cl[Ca]Cl": "s",
    "[Li]N([Si](C)(C)C)[Si](C)(C)C": "s",
    "C[Si](C)(C)N([Na])[Si](C)(C)C": "s",
    "CCO[Na]": "s",
}

# re-canonicalized
CHEMICAL_DENSITY = {
    "CC(C)CN1CCN2CCN(CC(C)C)P1N(CC(C)C)CC2": 964,
    "CN1CCCN(C)C1=O": 1060,
    "CCOC(=O)N=NC(=O)OCC": 1106,
    "CC[SiH](CC)CC": 728,
    "C[Si](C)(C)[N-][Si](C)(C)C.[Na+]": 900,
    "[SiH3]c1ccccc1": 877,
    "[NH4+].[NH4+].[S-2]": 997,
    "COc1ccc(CNc2cc(Oc3ccc(NC(=O)NC(=O)Cc4ccc(F)cc4)cc3F)ncn2)cc1": 1400,
    "[BH3]S(C)C": 801,
    "[BH3-][S+](C)C": 801,
    "B.CSC": 801,
    "CC[BH-](CC)CC.[Li+]": 890,
    "[Li][BH](CC)(CC)CC": 890,
    "Cl[SiH](Cl)Cl": 1342,
    "N#C[Zn]C#N": 1852,
    "CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]": 1200,
    "CC(C)C[Al]CC(C)C": 798,
    "[BH4-].[Li+]": 666,
    "CCN=C=N": 877,
    "CC1(C)OBOC1(C)C": 882,
    "CC(C)[SiH](C(C)C)C(C)C": 773,
    "CC(C)(C)O": 788,
    "CCN1CCOCC1": 899,
    "CN1CCOCC1": 920,
    "CC(C)COC(=O)Cl": 1040,
    "CC(C)(C)ON=O": 867,
    "CN(C)P(=O)(N(C)C)N(C)C": 1024,
    "O=S1(=O)CCCC1": 1260,
    "CCCCCCCC[N+](C)(CCCCCCCC)CCCCCCCC.[Cl-]": 884,
    "O=C(Cl)C(=O)Cl": 1478,
    "C[Si](C)(C)[N-][Si](C)(C)C.[Li+]": 860,
    "CCCCP(CCCC)CCCC": 820,
    "CC(C)(C)C(=O)Cl": 985,
    "CCOB(OCC)OCC": 858,
    "O=C(OC(=O)C(F)(F)F)C(F)(F)F": 1511,
    "CCOC(OCC)OCC": 891,
    "C[Si](C)(C)I": 1406,
    "CC(C)N=C=NC(C)C": 815,
    "CN1CCNCC1": 903,
    "CC(C)[O-].CC(C)[O-].CC(C)[O-].CC(C)[O-].[Ti+4]": 960,
    "CCN(CC)S(F)(F)F": 1220,
    "O=C(Cl)OCCCl": 1385,
    "COC(C)(C)OC": 850,
    "CC(C)=C(Cl)N(C)C": 1010,
    "COB(OC)OC": 915,
    "C[Si](C)(C)C#N": 793,
    "COC(OC)N(C)C": 897,
    "CS(=O)(=O)O": 1481,
    "COC(OC)OC": 967,
    "Cn1ccnc1": 1030,
    "CP(C)C": 735,
    "O=S(=O)(O)C(F)(F)F": 1700,
    "CC(C)(C)OCl": 958,
    "O=S(=O)(OS(=O)(=O)C(F)(F)F)C(F)(F)F": 1677,
    "CC(C)OB(OC(C)C)OC(C)C": 932,
    "CC(Cl)OC(=O)Cl": 1325,
    "S=C(Cl)Cl": 1508,
    "C[Si](C)(C)N=[N+]=[N-]": 876,
    "C[Si](C)(C)Br": 1160,
    "CCOP(OCC)OCC": 969,
    "C[Si](C)(C)OS(=O)(=O)C(F)(F)F": 1228,
    "CC(=O)C#N": 974,
    "COCCOCCN(CCOCCOC)CCOCCOC": 1011,
    "CCCCON=O": 1047,
    "O=P(Cl)(Cl)Oc1ccccc1": 1412,
    "CCB(CC)CC": 700,
    "CN(C=O)c1ccccc1": 1095,
    "CC1=CCC=CC1": 838,
    "O=CN1CCCCC1": 1019,
    "CN1CCCCC1": 816,
    "CN1CCN(C)C1=O": 1050,
    "CC1(C)CCCC(C)(C)N1": 830,
    "O=S(=O)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F": 1682,
    "O=C(Cl)c1ccccc1C(=O)Cl": 1409,
    "O=P(Cl)(Cl)c1ccccc1": 1319,
    "CCCCCON=O": 880,
    "OC(C(F)(F)F)C(F)(F)F": 1600,
    "O=S(=O)(Cl)c1ccccc1": 1384,
    "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21": 1362,
    "Fc1nc(F)nc(F)n1": 1574,
    "COP(OC)OC": 1052,
    "CCOC(C)(OCC)OCC": 887,
    "O=C(Cl)OC(Cl)(Cl)Cl": 1639,
    "CN(C)CCN(C)CCN(C)C": 830,
    "O=C=NS(=O)(=O)Cl": 1626,
    "c1ccc(OP(Oc2ccccc2)Oc2ccccc2)cc1": 1184,
    "CC(C)OC(=O)Cl": 892,
    "N=C(c1ccccc1)c1ccccc1": 1080,
    "COC(C)(OC)OC": 956,
    "C[C@H](N)c1ccccc1": 940,
    "CCCC[Sn](=O)CCCC": 1583,
    "COS(=O)(=O)C(F)(F)F": 1450,
    "CCOP(=O)(Cl)OCC": 1194,
    "N#CC(Cl)(Cl)Cl": 1440,
    "C[Si](C)(C)OCCO[Si](C)(C)C": 842,
    "CNc1ccc(Cl)cc1": 1169,
    "COc1ccc(CN)cc1": 1053,
    "CCOC(=O)[C@H](O)[C@@H](O)C(=O)OCC": 1204,
    "OCCC1CCCO1": 1051,
    "CCO[Ti](OCC)(OCC)OCC": 1088,
    "COCCN(CCOC)S(F)(F)F": 1200,
    "C1CCC(CNCC2CCCCC2)CC1": 870,
    "COB1OC(C)(C)C(C)(C)O1": 966,
    "CN(C)C(=NC(C)(C)C)N(C)C": 918,
    "CC(C)O[Ti](OC(C)C)(OC(C)C)OC(C)C": 960,
    "CC(C)(C)OC(=N)C(Cl)(Cl)Cl": 1222,
    "COP(=O)(OC)C(=[N+]=[N-])C(C)=O": 1280,
    "CCCP(=O)(O)OP(=O)(O)CCC": 1080,
    "NC(C1CCCCC1)C1CCCCC1": 900,
    "CCCC[Sn](CCCC)(CCCC)N=[N+]=[N-]": 1212,
    "CC(=N[Si](C)(C)C)O[Si](C)(C)C": 832,
    "CC(=O)OC=O": 1100,
    "CCN=C=NCCCN(C)C": 885,
    "C1CCC2=NCCCN2CC1": 1018,
    "[Li]C(C)CC": 769,
    "[Li]C": 732,
    "COC1CCCC1": 860,
    "CC(C)OC(=O)N=NC(=O)OC(C)C": 1027,
    "CC[Mg]Br": 1020,
    "CCOC(=O)OC(=O)OCC": 1101,
    "CN(C)C(=N)N(C)C": 918,
    "CN(C)C(OC(C)(C)C)OC(C)(C)C": 848,
    "CC(C)OC(=O)/N=N/C(=O)OC(C)C": 1030,
    "[2H]C(Cl)(Cl)Cl": 1500,
}


def split_neutral_fragment(reagents):
    reagents_neutral = set()
    reagents_charged = []
    for r in reagents:
        r_split = r.split(".")
        r_remaining = []
        for s in r_split:
            q = get_smiles_charge(s)
            if int(q) == 0:
                reagents_neutral.add(s)
            else:
                r_remaining.append(s)
        if len(r_remaining) > 0:
            r_remaining = ".".join(r_remaining)
            q = get_smiles_charge(r_remaining)
            if int(q) == 0:
                reagents_neutral.add(r_remaining)
            else:
                reagents_charged.append(r_remaining)
    return reagents_neutral, reagents_charged


def preprocess_reagents(reagents):
    """
    inputs: list of str, smiles
    outputs: list of str, smiles
    Rules:
    1. Neutral molecules are splitted from compounds
    2. Try to combine separated charged species
    3. Try to map Pd coordinated compounds with ligands
    4. Canonicalization using hardcoded rules
    """
    assert isinstance(reagents, list)
    for i in range(len(reagents)):
        reagents[i] = canonicalize_smiles_reagent_conv_rules(reagents[i])

    # Rule 1, split neutral
    reagents_neutral, reagents_charged = split_neutral_fragment(reagents)

    # Rule 2, combine charged, reagents_charged --> reagents_neutral
    # q for smiles in reagents_charged
    charges = [get_smiles_charge(s) for s in reagents_charged]
    # sanity check
    # check 1, total charge 0
    total_charge = sum(charges)
    if total_charge != 0:
        raise ProcessingError(
            f"preprocess_reagents(): total charge is not zero, q={str(total_charge)} \
                \nreagents: {reagents}, reagents_neutral: {reagents_neutral}, reagents_charged: {reagents_charged}"
        )
    if len(reagents_charged) > 0:
        reagents_neutral.add(canonicalize_smiles(".".join(reagents_charged)))
    reagents_neutral = list(reagents_neutral)

    # Rule 3, Canonicalization, replace using reagent_conv_rules.json
    res = set()
    for i in reagents_neutral:
        tmp = canonicalize_smiles_reagent_conv_rules(i)
        tmp1, tmp2 = split_neutral_fragment([tmp])
        if len(tmp2) != 0:
            sys.stderr.write(
                "preprocess_reagents(): error: charged fragment, s=" + str(reagents) + "\n"
            )
        for s in tmp1:
            res.add(s)

    return list(res)


def convert_smiles2cnt(s):
    """
    split smiles to {smiles:N}
    """
    res = {}
    for i in s.split("."):
        res[i] = res.get(i, 0) + 1
    return res


def get_solute_solvent(smiles):
    """
    split smiles into solvents and solutes
    Also try to split salts
    SOLUTES, solid, gas
    COMMON_SOLVENTS, liquid
    need special rules for NH3
    return solvents, solutes as dict of str:N
    """
    solutes = {}
    solvents = {}

    # split neutral
    neutral = {}
    r_remaining = {}
    for s in smiles.split("."):
        q = get_smiles_charge(s)
        if int(q) == 0:
            neutral[s] = neutral.get(s, 0) + 1
        else:
            r_remaining[s] = r_remaining.get(s, 0) + 1
    if len(r_remaining) > 0:
        charges = {s: get_smiles_charge(s) for s in r_remaining.keys()}
        # sanity check, total charge 0
        total_charge = 0
        for s, q in charges.items():
            total_charge += q * r_remaining[s]
        if total_charge != 0:
            raise ProcessingError(
                f"get_solute_solvent(): total charge is not zero, smiles={smiles}, q={total_charge}"
            )

        # check if only one cations or one anions
        num_cation = 0
        num_anions = 0
        for s, q in charges.items():
            if q > 0:
                num_cation += r_remaining[s]
            if q < 0:
                num_anions += r_remaining[s]
        if num_anions == 1 or num_cation == 1:
            s_cat = []
            for s in r_remaining:
                s_cat += [s] * r_remaining[s]
            s = ".".join(s_cat)
            s = canonicalize_smiles(s)
            # s = canonicalize_smiles_reagent_conv_rules(s)
            charged_split = {s: 1}
            solutes.update(charged_split)
            r_remaining = []

        # try to divide cations/anions into neutral groups
        charged_split = {}
        while len(r_remaining) > 0:
            match_found = False
            # find the largest negative q
            s_max = 0
            q_max = 0
            for s in r_remaining.keys():
                if charges[s] < q_max:
                    q_max = charges[s]
                    s_max = s
            # try to find cations
            for s in r_remaining.keys():
                q = charges[s]
                if -q_max * r_remaining[s_max] == q * r_remaining[s]:
                    d = math.gcd(abs(q_max), abs(q))
                    q_max /= d
                    q /= d
                    salt = ".".join([s_max] * int(abs(q)) + [s] * int(abs(q_max)))
                    charged_split[salt] = d
                    r_remaining.pop(s)
                    r_remaining.pop(s_max)
                    match_found = True
                    break
            if match_found:
                continue
            # no exact match
            for s in r_remaining.keys():
                q = charges[s]
                if (
                    -q_max * r_remaining[s_max] < q * r_remaining[s]
                    and q > 0
                    and abs(q_max * r_remaining[s_max]) % q == 0
                ):
                    n = int(abs(q_max * r_remaining[s_max] / q))
                    d = math.gcd(r_remaining[s_max], n)
                    salt = ".".join([s_max] * int(r_remaining[s_max] / d) + [s] * int(n / d))
                    charged_split[salt] = d
                    r_remaining[s] -= n
                    r_remaining.pop(s_max)
                    match_found = True
                    break
            if match_found:
                continue
            for s in r_remaining.keys():
                q = charges[s]
                if (
                    -q_max * r_remaining[s_max] > q * r_remaining[s]
                    and q > 0
                    and abs(q * r_remaining[s]) % q_max == 0
                ):
                    n = int(abs(q * r_remaining[s] / q_max))
                    d = math.gcd(r_remaining[s], n)
                    salt = ".".join([s_max] * int(n / d) + [s] * int(r_remaining[s] / d))
                    charged_split[salt] = d
                    r_remaining[s_max] -= n
                    r_remaining.pop(s)
                    match_found = True
                    break
            if match_found:
                continue
            # may need impovement
            raise ProcessingError(f"get_solute_solvent(): matching algo error for smiles={smiles}")

        # salts are solute, not really...
        charged_split_can = {}
        for s, v in charged_split.items():
            s = canonicalize_smiles(s)
            # s = canonicalize_smiles_reagent_conv_rules(s)
            charged_split_can[s] = v
        solutes.update(charged_split_can)

    for s, v in neutral.items():
        s = canonicalize_smiles(s)
        # s = canonicalize_smiles_reagent_conv_rules(s)
        if s in SOLUTES:
            solutes[s] = v
        elif s in COMMON_SOLVENTS_CANONICAL:  # jiannan's list -> pistachio's densitieis
            solvents[s] = v
        else:
            solutes[s] = v
            # phase = get_chemical_phase(s)
            # if phase == 's' or phase == 'g':
            #    solutes[s] = v
            # else:
            #    solvents[s] = v

    return solutes, solvents


special_conc = {
    #'CCO.O': 8.5, # ~50%v/v
    "O.[Na+].[Cl-]": 5.0,  # brine
    #'CCO.ClCCl': 4.0, #~50%v/v
    #'C1CCCO1.CC#N': 15.0, #~50%v/v
    "C[Si](C=[N+]=[N-])(C)C": 2.0,
    "CC(C[AlH]CC(C)C)C": 1.0,
    "COCCO[AlH2-]OCCOC.[Na+]": 3.5,  # ~70%wt in tulene
    "C1CC2CCCC(C1)B2": 0.5,  # in THF
    "CC[BH-](CC)CC.[Li+]": 1.0,  # super-hydride
    "CC[BH](CC)(CC)[Li]": 1.0,
    "Cl[Mg]C(C)C": 2.0,
    "[Li]CCCC": 1.0,
    "[Li]C(C)(C)C": 1.0,
}
