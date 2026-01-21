import math
import multiprocessing
import os
import time
from typing import List, Tuple, Dict, Optional, Union
from Bio.PDB import PDBParser, Superimposer, PDBIO, MMCIFParser, Residue
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio import pairwise2
from Bio.pairwise2 import Alignment

from multiprocessing import Pool

ALPHABETA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RES_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
# common_typos_disable
CCD_NAME_TO_ONE_LETTER: dict[str, str] = {
    "00C": "C",
    "01W": "X",
    "02K": "A",
    "03Y": "C",
    "07O": "C",
    "08P": "C",
    "0A0": "D",
    "0A1": "Y",
    "0A2": "K",
    "0A8": "C",
    "0AA": "V",
    "0AB": "V",
    "0AC": "G",
    "0AD": "G",
    "0AF": "W",
    "0AG": "L",
    "0AH": "S",
    "0AK": "D",
    "0AM": "A",
    "0AP": "C",
    "0AU": "U",
    "0AV": "A",
    "0AZ": "P",
    "0BN": "F",
    "0C": "C",
    "0CS": "A",
    "0DC": "C",
    "0DG": "G",
    "0DT": "T",
    "0FL": "A",
    "0G": "G",
    "0NC": "A",
    "0SP": "A",
    "0U": "U",
    "10C": "C",
    "125": "U",
    "126": "U",
    "127": "U",
    "128": "N",
    "12A": "A",
    "143": "C",
    "193": "X",
    "1AP": "A",
    "1MA": "A",
    "1MG": "G",
    "1PA": "F",
    "1PI": "A",
    "1PR": "N",
    "1SC": "C",
    "1TQ": "W",
    "1TY": "Y",
    "1X6": "S",
    "200": "F",
    "23F": "F",
    "23S": "X",
    "26B": "T",
    "2AD": "X",
    "2AG": "A",
    "2AO": "X",
    "2AR": "A",
    "2AS": "X",
    "2AT": "T",
    "2AU": "U",
    "2BD": "I",
    "2BT": "T",
    "2BU": "A",
    "2CO": "C",
    "2DA": "A",
    "2DF": "N",
    "2DM": "N",
    "2DO": "X",
    "2DT": "T",
    "2EG": "G",
    "2FE": "N",
    "2FI": "N",
    "2FM": "M",
    "2GT": "T",
    "2HF": "H",
    "2LU": "L",
    "2MA": "A",
    "2MG": "G",
    "2ML": "L",
    "2MR": "R",
    "2MT": "P",
    "2MU": "U",
    "2NT": "T",
    "2OM": "U",
    "2OT": "T",
    "2PI": "X",
    "2PR": "G",
    "2SA": "N",
    "2SI": "X",
    "2ST": "T",
    "2TL": "T",
    "2TY": "Y",
    "2VA": "V",
    "2XA": "C",
    "32S": "X",
    "32T": "X",
    "33X": "A",
    "3AH": "H",
    "3AR": "X",
    "3CF": "F",
    "3DA": "A",
    "3DR": "N",
    "3GA": "A",
    "3MD": "D",
    "3ME": "U",
    "3NF": "Y",
    "3QN": "K",
    "3TY": "X",
    "3XH": "G",
    "4AC": "N",
    "4BF": "Y",
    "4CF": "F",
    "4CY": "M",
    "4DP": "W",
    "4FB": "P",
    "4FW": "W",
    "4HT": "W",
    "4IN": "W",
    "4MF": "N",
    "4MM": "X",
    "4OC": "C",
    "4PC": "C",
    "4PD": "C",
    "4PE": "C",
    "4PH": "F",
    "4SC": "C",
    "4SU": "U",
    "4TA": "N",
    "4U7": "A",
    "56A": "H",
    "5AA": "A",
    "5AB": "A",
    "5AT": "T",
    "5BU": "U",
    "5CG": "G",
    "5CM": "C",
    "5CS": "C",
    "5FA": "A",
    "5FC": "C",
    "5FU": "U",
    "5HP": "E",
    "5HT": "T",
    "5HU": "U",
    "5IC": "C",
    "5IT": "T",
    "5IU": "U",
    "5MC": "C",
    "5MD": "N",
    "5MU": "U",
    "5NC": "C",
    "5PC": "C",
    "5PY": "T",
    "5SE": "U",
    "64T": "T",
    "6CL": "K",
    "6CT": "T",
    "6CW": "W",
    "6HA": "A",
    "6HC": "C",
    "6HG": "G",
    "6HN": "K",
    "6HT": "T",
    "6IA": "A",
    "6MA": "A",
    "6MC": "A",
    "6MI": "N",
    "6MT": "A",
    "6MZ": "N",
    "6OG": "G",
    "70U": "U",
    "7DA": "A",
    "7GU": "G",
    "7JA": "I",
    "7MG": "G",
    "8AN": "A",
    "8FG": "G",
    "8MG": "G",
    "8OG": "G",
    "9NE": "E",
    "9NF": "F",
    "9NR": "R",
    "9NV": "V",
    "A": "A",
    "A1P": "N",
    "A23": "A",
    "A2L": "A",
    "A2M": "A",
    "A34": "A",
    "A35": "A",
    "A38": "A",
    "A39": "A",
    "A3A": "A",
    "A3P": "A",
    "A40": "A",
    "A43": "A",
    "A44": "A",
    "A47": "A",
    "A5L": "A",
    "A5M": "C",
    "A5N": "N",
    "A5O": "A",
    "A66": "X",
    "AA3": "A",
    "AA4": "A",
    "AAR": "R",
    "AB7": "X",
    "ABA": "A",
    "ABR": "A",
    "ABS": "A",
    "ABT": "N",
    "ACB": "D",
    "ACL": "R",
    "AD2": "A",
    "ADD": "X",
    "ADX": "N",
    "AEA": "X",
    "AEI": "D",
    "AET": "A",
    "AFA": "N",
    "AFF": "N",
    "AFG": "G",
    "AGM": "R",
    "AGT": "C",
    "AHB": "N",
    "AHH": "X",
    "AHO": "A",
    "AHP": "A",
    "AHS": "X",
    "AHT": "X",
    "AIB": "A",
    "AKL": "D",
    "AKZ": "D",
    "ALA": "A",
    "N9P": "A",
    "ALC": "A",
    "ALM": "A",
    "ALN": "A",
    "ALO": "T",
    "ALQ": "X",
    "ALS": "A",
    "ALT": "A",
    "ALV": "A",
    "ALY": "K",
    "AN8": "A",
    "AP7": "A",
    "APE": "X",
    "APH": "A",
    "API": "K",
    "APK": "K",
    "APM": "X",
    "APP": "X",
    "AR2": "R",
    "AR4": "E",
    "AR7": "R",
    "ARG": "R",
    "ARM": "R",
    "ARO": "R",
    "ARV": "X",
    "AS": "A",
    "AS2": "D",
    "AS9": "X",
    "ASA": "D",
    "ASB": "D",
    "ASI": "D",
    "ASK": "D",
    "ASL": "D",
    "ASM": "X",
    "ASN": "N",
    "ASP": "D",
    "SOQ": "D",
    "ASQ": "D",
    "ASU": "N",
    "ASX": "B",
    "ATD": "T",
    "ATL": "T",
    "ATM": "T",
    "AVC": "A",
    "AVN": "X",
    "AYA": "A",
    "AZK": "K",
    "AZS": "S",
    "AZY": "Y",
    "B1F": "F",
    "B1P": "N",
    "B2A": "A",
    "B2F": "F",
    "B2I": "I",
    "B2V": "V",
    "B3A": "A",
    "B3D": "D",
    "B3E": "E",
    "B3K": "K",
    "B3L": "X",
    "B3M": "X",
    "B3Q": "X",
    "B3S": "S",
    "B3T": "X",
    "B3U": "H",
    "B3X": "N",
    "B3Y": "Y",
    "BB6": "C",
    "BB7": "C",
    "BB8": "F",
    "BB9": "C",
    "BBC": "C",
    "BCS": "C",
    "BE2": "X",
    "BFD": "D",
    "BG1": "S",
    "BGM": "G",
    "BH2": "D",
    "BHD": "D",
    "BIF": "F",
    "BIL": "X",
    "BIU": "I",
    "BJH": "X",
    "BLE": "L",
    "BLY": "K",
    "BMP": "N",
    "BMT": "T",
    "BNN": "F",
    "BNO": "X",
    "BOE": "T",
    "BOR": "R",
    "BPE": "C",
    "BRU": "U",
    "BSE": "S",
    "BT5": "N",
    "BTA": "L",
    "BTC": "C",
    "BTR": "W",
    "BUC": "C",
    "BUG": "V",
    "BVP": "U",
    "BZG": "N",
    "C": "C",
    "C1X": "K",
    "C25": "C",
    "C2L": "C",
    "C2S": "C",
    "C31": "C",
    "C32": "C",
    "C34": "C",
    "C36": "C",
    "C37": "C",
    "C38": "C",
    "C3Y": "C",
    "C42": "C",
    "C43": "C",
    "C45": "C",
    "C46": "C",
    "C49": "C",
    "C4R": "C",
    "C4S": "C",
    "C5C": "C",
    "C66": "X",
    "C6C": "C",
    "CAF": "C",
    "CAL": "X",
    "CAR": "C",
    "CAS": "C",
    "CAV": "X",
    "CAY": "C",
    "CB2": "C",
    "CBR": "C",
    "CBV": "C",
    "CCC": "C",
    "CCL": "K",
    "CCS": "C",
    "CDE": "X",
    "CDV": "X",
    "CDW": "C",
    "CEA": "C",
    "CFL": "C",
    "CG1": "G",
    "CGA": "E",
    "CGU": "E",
    "CH": "C",
    "CHF": "X",
    "CHG": "X",
    "CHP": "G",
    "CHS": "X",
    "CIR": "R",
    "CLE": "L",
    "CLG": "K",
    "CLH": "K",
    "CM0": "N",
    "CME": "C",
    "CMH": "C",
    "CML": "C",
    "CMR": "C",
    "CMT": "C",
    "CNU": "U",
    "CP1": "C",
    "CPC": "X",
    "CPI": "X",
    "CR5": "G",
    "CS0": "C",
    "CS1": "C",
    "CS3": "C",
    "CS4": "C",
    "CS8": "N",
    "CSA": "C",
    "CSB": "C",
    "CSD": "C",
    "CSE": "C",
    "CSF": "C",
    "CSI": "G",
    "CSJ": "C",
    "CSL": "C",
    "CSO": "C",
    "CSP": "C",
    "CSR": "C",
    "CSS": "C",
    "CSU": "C",
    "CSW": "C",
    "CSX": "C",
    "CSZ": "C",
    "CTE": "W",
    "CTG": "T",
    "CTH": "T",
    "CUC": "X",
    "CWR": "S",
    "CXM": "M",
    "CY0": "C",
    "CY1": "C",
    "CY3": "C",
    "CY4": "C",
    "CYA": "C",
    "CYD": "C",
    "CYF": "C",
    "CYG": "C",
    "CYJ": "X",
    "CYM": "C",
    "CYQ": "C",
    "CYR": "C",
    "CYS": "C",
    "060": "C",
    "CZ2": "C",
    "CZZ": "C",
    "D11": "T",
    "D1P": "N",
    "D3": "N",
    "D33": "N",
    "D3P": "G",
    "D3T": "T",
    "D4M": "T",
    "D4P": "X",
    "DA": "A",
    "DA2": "X",
    "DAB": "A",
    "DAH": "F",
    "DAL": "A",
    "DAR": "R",
    "DAS": "D",
    "DBB": "T",
    "DBM": "N",
    "DBS": "S",
    "DBU": "T",
    "DBY": "Y",
    "DBZ": "A",
    "DC": "C",
    "DC2": "C",
    "DCG": "G",
    "DCI": "X",
    "DCL": "X",
    "DCT": "C",
    "DCY": "C",
    "DDE": "H",
    "DDG": "G",
    "DDN": "U",
    "DDX": "N",
    "DFC": "C",
    "DFG": "G",
    "DFI": "X",
    "DFO": "X",
    "DFT": "N",
    "DG": "G",
    "DGH": "G",
    "DGI": "G",
    "DGL": "E",
    "DGN": "Q",
    "DHA": "S",
    "DHI": "H",
    "DHL": "X",
    "DHN": "V",
    "DHP": "X",
    "DHU": "U",
    "DHV": "V",
    "DI": "I",
    "DIL": "I",
    "DIR": "R",
    "DIV": "V",
    "DLE": "L",
    "DLS": "K",
    "DLY": "K",
    "DM0": "K",
    "DMH": "N",
    "DMK": "D",
    "DMT": "X",
    "DN": "N",
    "DNE": "L",
    "DNG": "L",
    "DNL": "K",
    "DNM": "L",
    "DNP": "A",
    "DNR": "C",
    "DNS": "K",
    "DOA": "X",
    "DOC": "C",
    "DOH": "D",
    "DON": "L",
    "DPB": "T",
    "DPH": "F",
    "DPL": "P",
    "DPP": "A",
    "DPQ": "Y",
    "DPR": "P",
    "DPY": "N",
    "DRM": "U",
    "DRP": "N",
    "DRT": "T",
    "DRZ": "N",
    "DSE": "S",
    "DSG": "N",
    "DSN": "S",
    "DSP": "D",
    "DT": "T",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DU": "U",
    "DVA": "V",
    "DXD": "N",
    "DXN": "N",
    "DYS": "C",
    "DZM": "A",
    "E": "A",
    "E1X": "A",
    "ECC": "Q",
    "EDA": "A",
    "EFC": "C",
    "EHP": "F",
    "EIT": "T",
    "ENP": "N",
    "ESB": "Y",
    "ESC": "M",
    "EXB": "X",
    "EXY": "L",
    "EY5": "N",
    "EYS": "X",
    "F2F": "F",
    "FA2": "A",
    "FA5": "N",
    "FAG": "N",
    "FAI": "N",
    "FB5": "A",
    "FB6": "A",
    "FCL": "F",
    "FFD": "N",
    "FGA": "E",
    "FGL": "G",
    "FGP": "S",
    "FHL": "X",
    "FHO": "K",
    "FHU": "U",
    "FLA": "A",
    "FLE": "L",
    "FLT": "Y",
    "FME": "M",
    "FMG": "G",
    "FMU": "N",
    "FOE": "C",
    "FOX": "G",
    "FP9": "P",
    "FPA": "F",
    "FRD": "X",
    "FT6": "W",
    "FTR": "W",
    "FTY": "Y",
    "FVA": "V",
    "FZN": "K",
    "G": "G",
    "G25": "G",
    "G2L": "G",
    "G2S": "G",
    "G31": "G",
    "G32": "G",
    "G33": "G",
    "G36": "G",
    "G38": "G",
    "G42": "G",
    "G46": "G",
    "G47": "G",
    "G48": "G",
    "G49": "G",
    "G4P": "N",
    "G7M": "G",
    "GAO": "G",
    "GAU": "E",
    "GCK": "C",
    "GCM": "X",
    "GDP": "G",
    "GDR": "G",
    "GFL": "G",
    "GGL": "E",
    "GH3": "G",
    "GHG": "Q",
    "GHP": "G",
    "GL3": "G",
    "GLH": "Q",
    "GLJ": "E",
    "GLK": "E",
    "GLM": "X",
    "GLN": "Q",
    "GLQ": "E",
    "GLU": "E",
    "GLX": "Z",
    "GLY": "G",
    "GLZ": "G",
    "GMA": "E",
    "GMS": "G",
    "GMU": "U",
    "GN7": "G",
    "GND": "X",
    "GNE": "N",
    "GOM": "G",
    "GPL": "K",
    "GS": "G",
    "GSC": "G",
    "GSR": "G",
    "GSS": "G",
    "GSU": "E",
    "GT9": "C",
    "GTP": "G",
    "GVL": "X",
    "H2U": "U",
    "H5M": "P",
    "HAC": "A",
    "HAR": "R",
    "HBN": "H",
    "HCS": "X",
    "HDP": "U",
    "HEU": "U",
    "HFA": "X",
    "HGL": "X",
    "HHI": "H",
    "HIA": "H",
    "HIC": "H",
    "HIP": "H",
    "HIQ": "H",
    "HIS": "H",
    "HL2": "L",
    "HLU": "L",
    "HMR": "R",
    "HOL": "N",
    "HPC": "F",
    "HPE": "F",
    "HPH": "F",
    "HPQ": "F",
    "HQA": "A",
    "HRG": "R",
    "HRP": "W",
    "HS8": "H",
    "HS9": "H",
    "HSE": "S",
    "HSL": "S",
    "HSO": "H",
    "HTI": "C",
    "HTN": "N",
    "HTR": "W",
    "HV5": "A",
    "HVA": "V",
    "HY3": "P",
    "HYP": "P",
    "HZP": "P",
    "I": "I",
    "I2M": "I",
    "I58": "K",
    "I5C": "C",
    "IAM": "A",
    "IAR": "R",
    "IAS": "D",
    "IC": "C",
    "IEL": "K",
    "IG": "G",
    "IGL": "G",
    "IGU": "G",
    "IIL": "I",
    "ILE": "I",
    "ILG": "E",
    "ILX": "I",
    "IMC": "C",
    "IML": "I",
    "IOY": "F",
    "IPG": "G",
    "IPN": "N",
    "IRN": "N",
    "IT1": "K",
    "IU": "U",
    "IYR": "Y",
    "IYT": "T",
    "IZO": "M",
    "JJJ": "C",
    "JJK": "C",
    "JJL": "C",
    "JW5": "N",
    "K1R": "C",
    "KAG": "G",
    "KCX": "K",
    "KGC": "K",
    "KNB": "A",
    "KOR": "M",
    "KPI": "K",
    "KST": "K",
    "KYQ": "K",
    "L2A": "X",
    "LA2": "K",
    "LAA": "D",
    "LAL": "A",
    "LBY": "K",
    "LC": "C",
    "LCA": "A",
    "LCC": "N",
    "LCG": "G",
    "LCH": "N",
    "LCK": "K",
    "LCX": "K",
    "LDH": "K",
    "LED": "L",
    "LEF": "L",
    "LEH": "L",
    "LEI": "V",
    "LEM": "L",
    "LEN": "L",
    "LET": "X",
    "LEU": "L",
    "LEX": "L",
    "LG": "G",
    "LGP": "G",
    "LHC": "X",
    "LHU": "U",
    "LKC": "N",
    "LLP": "K",
    "LLY": "K",
    "LME": "E",
    "LMF": "K",
    "LMQ": "Q",
    "LMS": "N",
    "LP6": "K",
    "LPD": "P",
    "LPG": "G",
    "LPL": "X",
    "LPS": "S",
    "LSO": "X",
    "LTA": "X",
    "LTR": "W",
    "LVG": "G",
    "LVN": "V",
    "LYF": "K",
    "LYK": "K",
    "LYM": "K",
    "LYN": "K",
    "LYR": "K",
    "LYS": "K",
    "PRK": "K",
    "NMK": "K",
    "LYX": "K",
    "LYZ": "K",
    "M0H": "C",
    "M1G": "G",
    "M2G": "G",
    "M2L": "K",
    "M2S": "M",
    "M30": "G",
    "M3L": "K",
    "M5M": "C",
    "MA": "A",
    "MA6": "A",
    "MA7": "A",
    "MAA": "A",
    "MAD": "A",
    "MAI": "R",
    "MBQ": "Y",
    "MBZ": "N",
    "MC1": "S",
    "MCG": "X",
    "MCL": "K",
    "MCS": "C",
    "MCY": "C",
    "MD3": "C",
    "MD6": "G",
    "MDH": "X",
    "MDR": "N",
    "MEA": "F",
    "MED": "M",
    "MEG": "E",
    "MEN": "N",
    "MEP": "U",
    "MEQ": "Q",
    "MET": "M",
    "MEU": "G",
    "MF3": "X",
    "MG1": "G",
    "MGG": "R",
    "MGN": "Q",
    "MGQ": "A",
    "MGV": "G",
    "MGY": "G",
    "MHL": "L",
    "MHO": "M",
    "MHS": "H",
    "MIA": "A",
    "MIS": "S",
    "MK8": "L",
    "ML3": "K",
    "MLE": "L",
    "MLL": "L",
    "MLY": "K",
    "MLZ": "K",
    "MME": "M",
    "MMO": "R",
    "MMT": "T",
    "MND": "N",
    "MNL": "L",
    "MNU": "U",
    "MNV": "V",
    "MOD": "X",
    "MP8": "P",
    "MPH": "X",
    "MPJ": "X",
    "MPQ": "G",
    "MRG": "G",
    "MSA": "G",
    "MSE": "M",
    "MSL": "M",
    "MSO": "M",
    "MSP": "X",
    "MT2": "M",
    "MTR": "T",
    "MTU": "A",
    "MTY": "Y",
    "MVA": "V",
    "N0A": "F",
    "N10": "S",
    "N2C": "X",
    "N5I": "N",
    "N5M": "C",
    "N6G": "G",
    "N7P": "P",
    "NA8": "A",
    "NAL": "A",
    "NAM": "A",
    "NB8": "N",
    "NBQ": "Y",
    "NC1": "S",
    "NCB": "A",
    "NCX": "N",
    "NCY": "X",
    "NDF": "F",
    "NDN": "U",
    "NEM": "H",
    "NEP": "H",
    "NF2": "N",
    "NFA": "F",
    "NHL": "E",
    "NIT": "X",
    "NIY": "Y",
    "NLE": "L",
    "NLN": "L",
    "NLO": "L",
    "NLP": "L",
    "NLQ": "Q",
    "NMC": "G",
    "NMM": "R",
    "NMS": "T",
    "NMT": "T",
    "NNH": "R",
    "NP3": "N",
    "NPH": "C",
    "NPI": "A",
    "NSK": "X",
    "NTY": "Y",
    "NVA": "V",
    "NYM": "N",
    "NYS": "C",
    "NZH": "H",
    "O12": "X",
    "O2C": "N",
    "O2G": "G",
    "OAD": "N",
    "OAS": "S",
    "OBF": "X",
    "OBS": "X",
    "OCS": "C",
    "OCY": "C",
    "ODP": "N",
    "OHI": "H",
    "OHS": "D",
    "OIC": "X",
    "OIP": "I",
    "OLE": "X",
    "OLT": "T",
    "OLZ": "S",
    "OMC": "C",
    "OMG": "G",
    "OMT": "M",
    "OMU": "U",
    "ONE": "U",
    "ONH": "A",
    "ONL": "X",
    "OPR": "R",
    "ORN": "A",
    "ORQ": "R",
    "OSE": "S",
    "OTB": "X",
    "OTH": "T",
    "OTY": "Y",
    "OXX": "D",
    "P": "G",
    "P1L": "C",
    "P1P": "N",
    "P2T": "T",
    "P2U": "U",
    "P2Y": "P",
    "P5P": "A",
    "PAQ": "Y",
    "PAS": "D",
    "PAT": "W",
    "PAU": "A",
    "PBB": "C",
    "PBF": "F",
    "PBT": "N",
    "PCA": "E",
    "PCC": "P",
    "PCE": "X",
    "PCS": "F",
    "PDL": "X",
    "PDU": "U",
    "PEC": "C",
    "PF5": "F",
    "PFF": "F",
    "PFX": "X",
    "PG1": "S",
    "PG7": "G",
    "PG9": "G",
    "PGL": "X",
    "PGN": "G",
    "PGP": "G",
    "PGY": "G",
    "PHA": "F",
    "PHD": "D",
    "PHE": "F",
    "PHI": "F",
    "PHL": "F",
    "PHM": "F",
    "PIV": "X",
    "PLE": "L",
    "PM3": "F",
    "PMT": "C",
    "POM": "P",
    "PPN": "F",
    "PPU": "A",
    "PPW": "G",
    "PQ1": "N",
    "PR3": "C",
    "PR5": "A",
    "PR9": "P",
    "PRN": "A",
    "PRO": "P",
    "PRS": "P",
    "PSA": "F",
    "PSH": "H",
    "PST": "T",
    "PSU": "U",
    "PSW": "C",
    "PTA": "X",
    "PTH": "Y",
    "PTM": "Y",
    "PTR": "Y",
    "PU": "A",
    "PUY": "N",
    "PVH": "H",
    "PVL": "X",
    "PYA": "A",
    "PYO": "U",
    "PYX": "C",
    "PYY": "N",
    "QMM": "Q",
    "QPA": "C",
    "QPH": "F",
    "QUO": "G",
    "R": "A",
    "R1A": "C",
    "R4K": "W",
    "RE0": "W",
    "RE3": "W",
    "RIA": "A",
    "RMP": "A",
    "RON": "X",
    "RT": "T",
    "RTP": "N",
    "S1H": "S",
    "S2C": "C",
    "S2D": "A",
    "S2M": "T",
    "S2P": "A",
    "S4A": "A",
    "S4C": "C",
    "S4G": "G",
    "S4U": "U",
    "S6G": "G",
    "SAC": "S",
    "SAH": "C",
    "SAR": "G",
    "SBL": "S",
    "SC": "C",
    "SCH": "C",
    "SCS": "C",
    "SCY": "C",
    "SD2": "X",
    "SDG": "G",
    "SDP": "S",
    "SEB": "S",
    "SEC": "A",
    "SEG": "A",
    "SEL": "S",
    "SEM": "S",
    "SEN": "S",
    "SEP": "S",
    "SER": "S",
    "SET": "S",
    "SGB": "S",
    "SHC": "C",
    "SHP": "G",
    "SHR": "K",
    "SIB": "C",
    "SLA": "P",
    "SLR": "P",
    "SLZ": "K",
    "SMC": "C",
    "SME": "M",
    "SMF": "F",
    "SMP": "A",
    "SMT": "T",
    "SNC": "C",
    "SNN": "N",
    "SOC": "C",
    "SOS": "N",
    "SOY": "S",
    "SPT": "T",
    "SRA": "A",
    "SSU": "U",
    "STY": "Y",
    "SUB": "X",
    "SUN": "S",
    "SUR": "U",
    "SVA": "S",
    "SVV": "S",
    "SVW": "S",
    "SVX": "S",
    "SVY": "S",
    "SVZ": "X",
    "SYS": "C",
    "T": "T",
    "T11": "F",
    "T23": "T",
    "T2S": "T",
    "T2T": "N",
    "T31": "U",
    "T32": "T",
    "T36": "T",
    "T37": "T",
    "T38": "T",
    "T39": "T",
    "T3P": "T",
    "T41": "T",
    "T48": "T",
    "T49": "T",
    "T4S": "T",
    "T5O": "U",
    "T5S": "T",
    "T66": "X",
    "T6A": "A",
    "TA3": "T",
    "TA4": "X",
    "TAF": "T",
    "TAL": "N",
    "TAV": "D",
    "TBG": "V",
    "TBM": "T",
    "TC1": "C",
    "TCP": "T",
    "TCQ": "Y",
    "TCR": "W",
    "TCY": "A",
    "TDD": "L",
    "TDY": "T",
    "TFE": "T",
    "TFO": "A",
    "TFQ": "F",
    "TFT": "T",
    "TGP": "G",
    "TH6": "T",
    "THC": "T",
    "THO": "X",
    "THR": "T",
    "NZC": "T",
    "THX": "N",
    "THZ": "R",
    "TIH": "A",
    "TLB": "N",
    "TLC": "T",
    "TLN": "U",
    "TMB": "T",
    "TMD": "T",
    "TNB": "C",
    "TNR": "S",
    "TOX": "W",
    "TP1": "T",
    "TPC": "C",
    "TPG": "G",
    "TPH": "X",
    "TPL": "W",
    "TPO": "T",
    "TPQ": "Y",
    "TQI": "W",
    "TQQ": "W",
    "TRF": "W",
    "TRG": "K",
    "TRN": "W",
    "TRO": "W",
    "TRP": "W",
    "TRQ": "W",
    "TRW": "W",
    "TRX": "W",
    "TS": "N",
    "TST": "X",
    "TT": "N",
    "TTD": "T",
    "TTI": "U",
    "TTM": "T",
    "TTQ": "W",
    "TTS": "Y",
    "TY1": "Y",
    "TY2": "Y",
    "TY3": "Y",
    "TY5": "Y",
    "TYB": "Y",
    "TYI": "Y",
    "TYJ": "Y",
    "TYN": "Y",
    "TYO": "Y",
    "TYQ": "Y",
    "TYR": "Y",
    "YNM": "Y",
    "TYS": "Y",
    "TYT": "Y",
    "TYU": "N",
    "TYW": "Y",
    "TYX": "X",
    "TYY": "Y",
    "TZB": "X",
    "TZO": "X",
    "U": "U",
    "U25": "U",
    "U2L": "U",
    "U2N": "U",
    "U2P": "U",
    "U31": "U",
    "U33": "U",
    "U34": "U",
    "U36": "U",
    "U37": "U",
    "U8U": "U",
    "UAR": "U",
    "UCL": "U",
    "UD5": "U",
    "UDP": "N",
    "UFP": "N",
    "UFR": "U",
    "UFT": "U",
    "UMA": "A",
    "UMP": "U",
    "UMS": "U",
    "UN1": "X",
    "UN2": "X",
    "UNK": "X",
    "UR3": "U",
    "URD": "U",
    "US1": "U",
    "US2": "U",
    "US3": "T",
    "US5": "U",
    "USM": "U",
    "VAD": "V",
    "VAF": "V",
    "VAL": "V",
    "VB1": "K",
    "VDL": "X",
    "VLL": "X",
    "VLM": "X",
    "VMS": "X",
    "VOL": "X",
    "X": "G",
    "X2W": "E",
    "X4A": "N",
    "XAD": "A",
    "XAE": "N",
    "XAL": "A",
    "XAR": "N",
    "XCL": "C",
    "XCN": "C",
    "XCP": "X",
    "XCR": "C",
    "XCS": "N",
    "XCT": "C",
    "XCY": "C",
    "XGA": "N",
    "XGL": "G",
    "XGR": "G",
    "XGU": "G",
    "XPR": "P",
    "XSN": "N",
    "XTH": "T",
    "XTL": "T",
    "XTR": "T",
    "XTS": "G",
    "XTY": "N",
    "XUA": "A",
    "XUG": "G",
    "XX1": "K",
    "Y": "A",
    "YCM": "C",
    "YG": "G",
    "YOF": "Y",
    "YRR": "N",
    "YYG": "G",
    "Z": "C",
    "Z01": "A",
    "ZAD": "A",
    "ZAL": "A",
    "ZBC": "C",
    "ZBU": "U",
    "ZCL": "F",
    "ZCY": "C",
    "ZDU": "U",
    "ZFB": "X",
    "ZGU": "G",
    "ZHP": "N",
    "ZTH": "T",
    "ZU0": "T",
    "ZZJ": "A",
    "DPN": "F",
    "MLU": "L",
    "ZAE": "F",
    "LE1": "V",
    "E95": "W",
    "1MH": "A",
    "LIG_E": "Y",
    "EME": "E",
    "5JP": "S",
    "E9M": "W",
    "F9D": "A",
    "004": "F",
    "4PQ": "W",
    "HOX": "F",
    "2L5": "F",
    "CCJ": "C",
    "DA2": "R",
    "GME": "E",
    "2GX": "F",
    "6CV": "F",
    "6DU": "F",
}
# common_typos_enable
# pyformat: enable


class AtomPair:
    def __init__(
        self,
        pre_chain_id: str,
        ref_chain_id: str,
        pred_sequence: str,
        ref_sequence: str,
        pred_res: list[Residue],
        ref_res: list[Residue],
    ):
        self.rmsd = 0
        self.pre_chain_id = pre_chain_id
        self.ref_chain_id = ref_chain_id
        self.pred_sequence = pred_sequence
        self.ref_sequence = ref_sequence
        self.pred_res = pred_res
        self.ref_res = ref_res

        self.pred_atoms: list[Atom] = []
        self.ref_atoms: list[Atom] = []

        self._align_mached_chains()

    def _align_mached_chains(self):
        pred_res_index = -1
        ref_res_index = -1
        for i, res_name in enumerate(self.pred_sequence):
            if res_name != "-":
                pred_res_index += 1

            if self.ref_sequence[i] != "-":
                ref_res_index += 1

            if res_name == "-" and self.ref_sequence[i] == "-":
                continue

            if res_name != self.ref_sequence[i]:
                continue
            assert self.pred_res[pred_res_index].resname == self.ref_res[ref_res_index].resname
            for atom_name, atom in self.pred_res[pred_res_index].child_dict.items():
                if atom_name not in self.ref_res[ref_res_index].child_dict:
                    continue
                atom.parent._id = ("", pred_res_index + 1, "")
                atom.parent._reset_full_id()

                ref_atom = self.ref_res[ref_res_index].child_dict[atom_name]
                ref_atom.parent._id = ("", ref_res_index + 1, "")
                ref_atom.parent.parent._id = self.pre_chain_id
                ref_atom.parent._reset_full_id()

                self.pred_atoms.append(atom)
                self.ref_atoms.append(ref_atom)


class AlignPair:
    def __init__(
        self,
        reference_structure_file: Path | str,
        predict_structure_file: Path | str,
        binder_chain_id: str | None = None,
        backbone_atoms: List[str] = ["N", "C", "CA"],
    ):

        # init file path
        self.reference_structure_file = reference_structure_file
        if isinstance(reference_structure_file, str):
            self.reference_structure_file = Path(reference_structure_file)
        self.predict_structure_file = predict_structure_file
        if isinstance(predict_structure_file, str):
            self.predict_structure_file = Path(predict_structure_file)

        self.id = self.reference_structure_file.stem
        self.binder_chain_id = binder_chain_id
        # init structure
        reference_structure, predict_structure = self.__init_structure()
        self.predict_structure: Model = predict_structure
        ## restructure reference structure
        self.reference_structure: Model = reference_structure
        aligned_atom_pairs_combo, aligned_atom_pairs_ca_imposer = self._align_chains()
        self.aligned_atom_pairs_ca_imposer = aligned_atom_pairs_ca_imposer

        self.backbone_atoms = backbone_atoms
        self._split_aligned_atoms(aligned_atom_pairs_combo)

        if self.binder_chain_id is None:
            min_chain_len = float("inf")
            for chain in self.predict_structure:
                if len(chain) < min_chain_len:
                    min_chain_len = len(chain)
                    self.binder_chain_id = chain.id

    def __init_structure(self) -> Tuple[Model, Model]:
        parser = PDBParser(QUIET=True)
        reference_structure = parser.get_structure("reference", self.reference_structure_file)[0]
        if self.predict_structure_file.suffix in [".cif", ".mmcif"]:
            parser = MMCIFParser(QUIET=True)
        predict_structure = parser.get_structure("predict", self.predict_structure_file)[0]
        return reference_structure, predict_structure

    def _find_all_valid_combinations(self, str_lists: list[list[str]]) -> list[list[str]]:

        n = len(str_lists)
        result = []

        def backtrack(index, current_chars: list[str]):
            if index == n:
                result.append(current_chars.copy())
                return

            for char in str_lists[index]:
                if char not in current_chars:
                    current_chars.append(char)
                    backtrack(index + 1, current_chars)
                    current_chars.pop()

        backtrack(0, [])

        return result

    def _calc_combo_ca_imposer(self, aligned_atom_pairs_combo: list[AtomPair]) -> Superimposer:

        all_atoms_ca_ref = []
        all_atoms_ca_predict = []
        for atom_pair in aligned_atom_pairs_combo:
            atoms_ca_ref = [atom for atom in atom_pair.ref_atoms if atom.get_full_id()[4][0] == "CA"]
            all_atoms_ca_ref.extend(atoms_ca_ref)
            atoms_ca_predict = [atom for atom in atom_pair.pred_atoms if atom.get_full_id()[4][0] == "CA"]
            all_atoms_ca_predict.extend(atoms_ca_predict)

        super_imposer = Superimposer()
        super_imposer.set_atoms(all_atoms_ca_ref, all_atoms_ca_predict)
        return super_imposer

    def _align_chains(self) -> Tuple[list[AtomPair], Superimposer]:
        ref_structure_res: dict[str, list[Residue]] = {}
        ref_structure_seqs: dict[str, str] = {}

        predict_structure_seqs: dict[str, str] = {}
        predict_structure_res: dict[str, list[Residue]] = {}
        for chain in self.predict_structure:
            chain_res_name = []
            chain_res = []
            for res in chain:
                if res.resname in CCD_NAME_TO_ONE_LETTER:
                    chain_res_name.append(CCD_NAME_TO_ONE_LETTER[res.resname])
                    chain_res.append(res)
                else:
                    chain_res_name.append(f"({res.resname})")
                    chain_res.append(res)
            predict_structure_seqs[chain.id] = "".join(chain_res_name)
            predict_structure_res[chain.id] = chain_res

        for chain in self.reference_structure:
            chain_res = []
            chain_res_name = []
            for res in chain:
                if res.resname == "UNK":
                    continue
                if res.resname == "HOH":
                    continue

                if res.resname in CCD_NAME_TO_ONE_LETTER:
                    chain_res.append(res)
                    chain_res_name.append(CCD_NAME_TO_ONE_LETTER[res.resname])
                else:
                    ref_structure_res[chain.id] = chain_res
                    ref_structure_seqs[chain.id] = "".join(chain_res_name)
                    res._id = (res._id[0], 1, res._id[2])
                    ref_structure_res[res.resname] = [res]
                    ref_structure_seqs[res.resname] = f"({res.resname})"
                    break
            ref_structure_res[chain.id] = chain_res
            ref_structure_seqs[chain.id] = "".join(chain_res_name)

        aligned_pairs: dict[str, list[AtomPair]] = {}
        matched_flag: dict[str, list[str]] = {}

        for pre_chain_id, pre_chain_seq in predict_structure_seqs.items():

            for chain_id, chain_seq in ref_structure_seqs.items():
                alignments: Alignment = pairwise2.align.localxx(pre_chain_seq, chain_seq)[0]
                score = alignments.score
                success_rate = score / max(len(chain_seq), len(pre_chain_seq))
                if success_rate < 0.7:
                    continue
                atom_pair = AtomPair(
                    pre_chain_id,
                    chain_id,
                    alignments.seqA,
                    alignments.seqB,
                    predict_structure_res[pre_chain_id],
                    ref_structure_res[chain_id],
                )

                if pre_chain_id not in aligned_pairs:
                    aligned_pairs[pre_chain_id] = []
                    matched_flag[pre_chain_id] = []

                aligned_pairs[pre_chain_id].append(atom_pair)
                matched_flag[pre_chain_id].append(chain_id)

        aligned_chain_combos = self._find_all_valid_combinations(list(matched_flag.values()))
        best_aligned_atom_pairs_combo = []
        best_aligned_atom_pairs_ca_imposer = Superimposer()
        best_rmsd = float("inf")
        verbose_dict: dict[str, float] = {}
        for combo in aligned_chain_combos:
            aligned_atom_pairs_combo = []
            for (pre_chain_id, aligned_atom_pairs), aligned_chain_id in zip(aligned_pairs.items(), combo):
                for atom_pair in aligned_atom_pairs:
                    if atom_pair.ref_chain_id != aligned_chain_id:
                        continue
                    aligned_atom_pairs_combo.append(atom_pair)
            current_combo_ca_imposer = self._calc_combo_ca_imposer(aligned_atom_pairs_combo)
            current_combo_rmsd = current_combo_ca_imposer.rms
            verbose_dict["".join(combo)] = current_combo_rmsd
            if current_combo_rmsd < best_rmsd:
                best_rmsd = current_combo_rmsd
                best_aligned_atom_pairs_combo = aligned_atom_pairs_combo
                best_aligned_atom_pairs_ca_imposer = current_combo_ca_imposer

        return best_aligned_atom_pairs_combo, best_aligned_atom_pairs_ca_imposer

    def _split_aligned_atoms(self, aligned_atom_pairs_combo: list[AtomPair]):

        self.pred_atoms: list[Atom] = []
        self.pred_chain_atoms: dict[str, list[Atom]] = {}

        self.ref_atoms: list[Atom] = []
        self.ref_chain_atoms: dict[str, list[Atom]] = {}

        self.pred_backbone_atoms: list[Atom] = []
        self.pred_chain_backbone_atoms: dict[str, list[Atom]] = {}

        self.ref_backbone_atoms: list[Atom] = []
        self.ref_chain_backbone_atoms: dict[str, list[Atom]] = {}

        self.pred_ca_atoms: list[Atom] = []
        self.pred_chain_ca_atoms: dict[str, list[Atom]] = {}

        self.ref_ca_atoms: list[Atom] = []
        self.ref_chain_ca_atoms: dict[str, list[Atom]] = {}

        self.pred_uaa_atoms: list[Atom] = []
        self.pred_chain_uaa_atoms: dict[str, list[Atom]] = {}

        self.ref_uaa_atoms: list[Atom] = []
        self.ref_chain_uaa_atoms: dict[str, list[Atom]] = {}

        for atom_pair in aligned_atom_pairs_combo:
            pred_ca_atoms = []
            ref_ca_atoms = []
            pred_backbone_atoms = []
            ref_backbone_atoms = []
            pred_uaa_atoms = []
            ref_uaa_atoms = []
            for pred_atom, ref_atom in zip(atom_pair.pred_atoms, atom_pair.ref_atoms):
                self.pred_atoms.append(pred_atom)
                self.ref_atoms.append(ref_atom)
                if pred_atom.get_full_id()[4][0] == "CA":
                    pred_ca_atoms.append(pred_atom)
                    ref_ca_atoms.append(ref_atom)

                if pred_atom.get_full_id()[4][0] in self.backbone_atoms:
                    pred_backbone_atoms.append(pred_atom)
                    ref_backbone_atoms.append(ref_atom)

                if pred_atom.get_parent().resname not in RES_1TO3.values():
                    pred_uaa_atoms.append(pred_atom)
                    ref_uaa_atoms.append(ref_atom)

            self.pred_ca_atoms.extend(pred_ca_atoms)
            self.ref_ca_atoms.extend(ref_ca_atoms)
            self.pred_backbone_atoms.extend(pred_backbone_atoms)
            self.ref_backbone_atoms.extend(ref_backbone_atoms)
            self.pred_uaa_atoms.extend(pred_uaa_atoms)
            self.ref_uaa_atoms.extend(ref_uaa_atoms)

            self.pred_chain_ca_atoms[atom_pair.pre_chain_id] = pred_ca_atoms
            self.ref_chain_ca_atoms[atom_pair.pre_chain_id] = ref_ca_atoms
            self.pred_chain_backbone_atoms[atom_pair.pre_chain_id] = pred_backbone_atoms
            self.ref_chain_backbone_atoms[atom_pair.pre_chain_id] = ref_backbone_atoms
            self.pred_chain_uaa_atoms[atom_pair.pre_chain_id] = pred_uaa_atoms
            self.ref_chain_uaa_atoms[atom_pair.pre_chain_id] = ref_uaa_atoms
            self.pred_chain_atoms[atom_pair.pre_chain_id] = atom_pair.pred_atoms
            self.ref_chain_atoms[atom_pair.pre_chain_id] = atom_pair.ref_atoms


class Metrics:
    def __init__(
        self,
        reference_structure_file: Path | str,
        predict_structure_file: Path | str,
        binder_chain_id: str | None = None,
        backbone_atoms: List[str] = ["N", "C", "CA"],
        cutoff=5.0,
    ):

        self.aligned_pair: AlignPair = AlignPair(
            reference_structure_file, predict_structure_file, binder_chain_id, backbone_atoms
        )
        self.id = self.aligned_pair.id
        self.cutoff = cutoff

        self.all_atom_rmsd = 0
        self.chain_all_atom_rmsd = {}

        self.chain_backbone_rmsd = {}
        self.backbone_rmsd = 0

        self.chain_ca_rmsd = {}
        self.ca_rmsd = 0

        self.chain_all_atom_rmsd = {}
        self.all_atom_rmsd = 0

        self.chain_uaa_atom_rmsd = {}
        self.uaa_atom_rmsd = 0

        self.aligned_pair.aligned_atom_pairs_ca_imposer.apply(self.aligned_pair.pred_atoms)

        self.fnat = self._calc_fnat()
        self._calc_rmsd()

    def __repr__(self) -> str:
        repr_str = f"Metrics for {self.id}:\n"
        repr_str += f"  Fnat: {self.fnat:.4f}\n"
        repr_str += f"  Overall RMSD: {self.rmsd:.4f} Å\n"
        repr_str += f"  CA RMSD: {self.ca_rmsd:.4f} Å\n"
        repr_str += f"  Backbone RMSD: {self.backbone_rmsd:.4f} Å\n"
        repr_str += f"  All Atom RMSD: {self.all_atom_rmsd:.4f} Å\n"
        repr_str += f"  UAA Atom RMSD: {self.uaa_atom_rmsd:.4f} Å\n"
        repr_str += "  Chain-wise RMSD:\n"
        for chain_id in self.chain_ca_rmsd:
            repr_str += f"    Chain {chain_id} CA RMSD: {self.chain_ca_rmsd[chain_id]:.4f} Å\n"
            repr_str += f"    Chain {chain_id} Backbone RMSD: {self.chain_backbone_rmsd[chain_id]:.4f} Å\n"
            repr_str += f"    Chain {chain_id} All Atom RMSD: {self.chain_all_atom_rmsd[chain_id]:.4f} Å\n"
            repr_str += f"    Chain {chain_id} UAA Atom RMSD: {self.chain_uaa_atom_rmsd[chain_id]:.4f} Å\n"
        return repr_str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "fnat": self.fnat,
            "rmsd": self.rmsd,
            "ca_rmsd": self.ca_rmsd,
            "backbone_rmsd": self.backbone_rmsd,
            "all_atom_rmsd": self.all_atom_rmsd,
            "uaa_atom_rmsd": self.uaa_atom_rmsd,
            # "chain_ca_rmsd": self.chain_ca_rmsd,
            # "chain_backbone_rmsd": self.chain_backbone_rmsd,
            # "chain_all_atom_rmsd": self.chain_all_atom_rmsd,
            # "chain_uaa_atom_rmsd": self.chain_uaa_atom_rmsd,
            **self.binder_rmsd(),
        }

    def binder_rmsd(self):

        return {
            "binder_id": self.aligned_pair.binder_chain_id,
            "binder_ca_rmsd": self.chain_ca_rmsd[self.aligned_pair.binder_chain_id],
            "binder_backbone_rmsd": self.chain_backbone_rmsd[self.aligned_pair.binder_chain_id],
            "binder_all_atom_rmsd": self.chain_all_atom_rmsd[self.aligned_pair.binder_chain_id],
            "binder_uaa_atom_rmsd": self.chain_uaa_atom_rmsd[self.aligned_pair.binder_chain_id],
        }

    def _calculate_contacts(self, atoms: dict[str, list[Atom]], binder_chain_id: str) -> set[Tuple[int, int]]:
        """
        计算结构中两个链之间的接触，使用 CA 原子之间的距离。

        :param structure: Biopython structure 对象.
        :param binder_chain_id: 配体链的 ID (例如 "A").
        :param cutoff: 定义接触的距离阈值 (默认 5.0 Å).
        :return: 接触对集合，格式为 (残基编号1, 残基编号2).
        """
        contacts = set()

        for atom1 in atoms[binder_chain_id]:
            for chain_id in atoms.keys():
                if chain_id == binder_chain_id:
                    continue
                chain_2 = atoms[chain_id]
                for atom2 in chain_2:
                    if atom1 - atom2 < self.cutoff:
                        contacts.add(
                            (atom1.get_parent().get_id()[1], f"{atom2.get_full_id()[2]}{atom2.get_full_id()[3][1]}")
                        )

        return contacts

    def _calc_fnat(self):
        if len(self.aligned_pair.reference_structure) < 2:
            return -1.0

        reference_contacts = self._calculate_contacts(
            self.aligned_pair.ref_chain_atoms, self.aligned_pair.binder_chain_id
        )

        if not reference_contacts:
            return 0.0

        predicted_contacts = self._calculate_contacts(
            self.aligned_pair.pred_chain_atoms, self.aligned_pair.binder_chain_id
        )

        common_contacts = reference_contacts.intersection(predicted_contacts)
        return len(common_contacts) / len(reference_contacts)

    def _calc_ca_rmsd(self):
        for chain_id, pred_ca_atoms in self.aligned_pair.pred_chain_ca_atoms.items():
            chain_ca_atoms_sd = [
                (x - y) * (x - y) for x, y in zip(pred_ca_atoms, self.aligned_pair.ref_chain_ca_atoms[chain_id])
            ]
            chain_ca_rmsd = pow(sum(chain_ca_atoms_sd) / len(chain_ca_atoms_sd), 0.5)
            self.chain_ca_rmsd[chain_id] = chain_ca_rmsd

        ca_atoms_sd = [
            (x - y) * (x - y) for x, y in zip(self.aligned_pair.pred_ca_atoms, self.aligned_pair.ref_ca_atoms)
        ]
        self.ca_rmsd = pow(sum(ca_atoms_sd) / len(ca_atoms_sd), 0.5)

    def _calc_backbone_rmsd(self):
        for chain_id, pred_backbone_atoms in self.aligned_pair.pred_chain_backbone_atoms.items():
            chain_backbone_atoms_sd = [
                (x - y) * (x - y)
                for x, y in zip(pred_backbone_atoms, self.aligned_pair.ref_chain_backbone_atoms[chain_id])
            ]
            chain_backbone_rmsd = pow(sum(chain_backbone_atoms_sd) / len(chain_backbone_atoms_sd), 0.5)
            self.chain_backbone_rmsd[chain_id] = chain_backbone_rmsd

        backbone_atoms_sd = [
            (x - y) * (x - y)
            for x, y in zip(self.aligned_pair.pred_backbone_atoms, self.aligned_pair.ref_backbone_atoms)
        ]
        self.backbone_rmsd = pow(sum(backbone_atoms_sd) / len(backbone_atoms_sd), 0.5)

    def _calc_uaa_rmsd(self):
        for chain_id, pred_uaa_atoms in self.aligned_pair.pred_chain_uaa_atoms.items():
            chain_uaa_atoms_sd = [
                (x - y) * (x - y) for x, y in zip(pred_uaa_atoms, self.aligned_pair.ref_chain_uaa_atoms[chain_id])
            ]
            if len(chain_uaa_atoms_sd) > 0:
                chain_uaa_rmsd = pow(sum(chain_uaa_atoms_sd) / len(chain_uaa_atoms_sd), 0.5)
                self.chain_uaa_atom_rmsd[chain_id] = chain_uaa_rmsd
            else:
                self.chain_uaa_atom_rmsd[chain_id] = 0.0

        uaa_atoms_sd = [
            (x - y) * (x - y) for x, y in zip(self.aligned_pair.pred_uaa_atoms, self.aligned_pair.ref_uaa_atoms)
        ]
        if len(uaa_atoms_sd) > 0:
            self.uaa_atom_rmsd = pow(sum(uaa_atoms_sd) / len(uaa_atoms_sd), 0.5)
        else:
            self.uaa_atom_rmsd = 0.0

    def _calc_all_atoms_rmsd(self):
        for chain_id, pred_atoms in self.aligned_pair.pred_chain_atoms.items():
            chain_all_atoms_sd = [
                (x - y) * (x - y) for x, y in zip(pred_atoms, self.aligned_pair.ref_chain_atoms[chain_id])
            ]
            chain_all_atom_rmsd = pow(sum(chain_all_atoms_sd) / len(chain_all_atoms_sd), 0.5)
            self.chain_all_atom_rmsd[chain_id] = chain_all_atom_rmsd

        all_atoms_sd = [(x - y) * (x - y) for x, y in zip(self.aligned_pair.pred_atoms, self.aligned_pair.ref_atoms)]
        self.all_atom_rmsd = pow(sum(all_atoms_sd) / len(all_atoms_sd), 0.5)

    def _calc_rmsd(self):
        self.rmsd = self.aligned_pair.aligned_atom_pairs_ca_imposer.rms
        self._calc_all_atoms_rmsd()
        self._calc_ca_rmsd()
        self._calc_backbone_rmsd()
        self._calc_uaa_rmsd()


class MetricsRunner:
    def __init__(
        self,
        backbone_atoms: List[str] = ["N", "C", "CA"],
        binder_chain_id: str | None = None,
        cutoff=5.0,
        use_multi_process=False,
    ):
        self.backbone_atoms = backbone_atoms
        self.binder_chain_id = binder_chain_id
        self.cutoff = cutoff
        self.use_multi_process = use_multi_process
        self.cpu_num = multiprocessing.cpu_count()

    def run_batch(
        self,
        reference_structure_files: List[Path | str],
        predict_structure_files: List[Path | str],
        output_csv_file: Path = None,
        save: bool = True,
    ) -> pd.DataFrame:
        metrics_list = []

        if not self.use_multi_process:

            for ref_file, pre_file in zip(reference_structure_files, predict_structure_files):
                metrics = Metrics(
                    ref_file,
                    pre_file,
                    binder_chain_id=self.binder_chain_id,
                    backbone_atoms=self.backbone_atoms,
                    cutoff=self.cutoff,
                )
                metrics_list.append(metrics.as_dict())

        else:
            with multiprocessing.Pool(self.cpu_num) as pool:
                metrics_list = pool.starmap(
                    self.run_single_eval,
                    [
                        (ref_file, pre_file)
                        for ref_file, pre_file in zip(reference_structure_files, predict_structure_files)
                    ],
                )

            metrics_list = [m for m in metrics_list if m is not None]

        df = pd.DataFrame(metrics_list)
        if save and output_csv_file is not None:
            df.to_csv(output_csv_file, index=False)
        return df

    def run_af3_dataset_eval(
        self, data_set_dir: Path, data_tag: str, output_csv_file: Path = None, save: bool = False
    ) -> pd.DataFrame:
        metrics_list = []
        ref_pdb_files = list(data_set_dir.rglob("*.pdb"))
        if not self.use_multi_process:
            for ref_pdb_file in ref_pdb_files:
                data_id = ref_pdb_file.stem.lower()
                pre_cif_file = ref_pdb_file.parent / f"{data_tag}/{data_id}_model.cif"
                if not pre_cif_file.exists():
                    continue
                metrics = Metrics(
                    ref_pdb_file,
                    pre_cif_file,
                    binder_chain_id=self.binder_chain_id,
                    backbone_atoms=self.backbone_atoms,
                    cutoff=self.cutoff,
                )
                metrics_list.append(metrics.as_dict())
        else:
            with multiprocessing.Pool() as pool:
                metrics_list = pool.starmap(
                    Metrics,
                    [
                        (ref_pdb_file, ref_pdb_file.parent / f"{data_tag}/{ref_pdb_file.stem}_model.cif")
                        for ref_pdb_file in ref_pdb_files
                    ],
                )

            metrics_list = [m.as_dict() for m in metrics_list if m is not None]

        df = pd.DataFrame(metrics_list)
        if save and output_csv_file is not None:
            df.to_csv(output_csv_file, index=False)
        return df

    def run_boltz_dataset_eval(self, data_set_dir: Path, model_out_dir: Path) -> list[dict]:
        metrics_list = []
        ref_pdb_files = list(data_set_dir.rglob("*.pdb"))
        if not self.use_multi_process:
            for ref_pdb_file in ref_pdb_files:
                data_id = ref_pdb_file.stem.lower()
                pre_cif_file = model_out_dir / f"{data_id.upper()}/{data_id.upper()}_model_0.pdb"
                if not pre_cif_file.exists():
                    continue
                metrics = Metrics(
                    ref_pdb_file,
                    pre_cif_file,
                    binder_chain_id=self.binder_chain_id,
                    backbone_atoms=self.backbone_atoms,
                    cutoff=self.cutoff,
                )
                metrics_list.append(metrics.as_dict())
        else:
            with multiprocessing.Pool() as pool:
                metrics_list = pool.starmap(
                    Metrics,
                    [
                        (
                            ref_pdb_file,
                            model_out_dir / f"{ref_pdb_file.stem.upper()}/{ref_pdb_file.stem.upper()}_model_0.pdb",
                        )
                        for ref_pdb_file in ref_pdb_files
                    ],
                )

            metrics_list = [m.as_dict() for m in metrics_list if m is not None]

        return metrics_list

    def run_single_eval(self, ref_pdb_file: Path, pre_cif_file: Path) -> dict:
        metrics = Metrics(
            ref_pdb_file,
            pre_cif_file,
            binder_chain_id=self.binder_chain_id,
            backbone_atoms=self.backbone_atoms,
            cutoff=self.cutoff,
        )

        return metrics.as_dict()
