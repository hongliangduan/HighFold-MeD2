# data process utils. file opts and simple calc.
import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from boltz_finetune.common.config import UPPER_LETTERS_START
from boltz.data.types import Manifest

ACID2RES_DICT = {
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
ACID_NAME_2RES_DICT = {
    "ALANINE": "ALA",
    "ARGININE": "ARG",
    "ASPARAGINE": "ASN",
    "ASPARTIC": "ASP",
    "CYSTEINE": "CYS",
    "GLUTAMINE": "GLN",
    "GLUTAMIC": "GLU",
    "GLYCINE": "GLY",
    "HISTIDINE": "HIS",
    "ISOLEUCINE": "ILE",
    "LEUCINE": "LEU",
    "LYSINE": "LYS",
    "METHIONINE": "MET",
    "PHENYLALANINE": "PHE",
    "PROLINE": "PRO",
    "SERINE": "SER",
    "THREONINE": "THR",
    "TRYPTOPHAN": "TRP",
    "TYROSINE": "TYR",
    "VALINE": "VAL",
}
RES2ACID_DICT = dict([val, key] for key, val in ACID2RES_DICT.items())


class ChainType(Enum):
    Protein = 1
    Ligand = 2
    RNA = 3
    DNA = 4


CHAIN_TYPE_DICT = {ChainType.Protein: "protein", ChainType.Ligand: "ligand", ChainType.RNA: "rna", ChainType.DNA: "dna"}


def is_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def get_dir_files(data_dir: Path, file_tag: str) -> List[Path]:

    return list(data_dir.rglob(f"*.{file_tag}"))


def dump_data(data: Manifest, file_path: Path) -> bool:
    if len(data.records) > 0:

        if not file_path.exists():
            os.makedirs(file_path.parent, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(data, file)

        if file_path.exists():
            return True

    return False


def load_data(file_path: Path) -> Manifest:
    if file_path.exists():
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    return None


def load_yaml_as_dict(yaml_file: Path) -> Dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_value = yaml.safe_load(f)
    return yaml_value


def read_fasta_file(path: Path) -> List[Tuple[str, str]]:

    fasta_string = path.read_text()
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def mock_sequence_id(index: int) -> str:
    return chr(index + UPPER_LETTERS_START)
