import re
from enum import Enum
from pathlib import Path
from typing import Any, List, TypeAlias, Final
from rdkit import Chem
import pickle
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from loguru import logger

from boltz_finetune.common.config import MAZ_TEMPLATES_NUM
from boltz_finetune.utils.data_utils import (
    ACID_NAME_2RES_DICT,
    RES2ACID_DICT,
    CHAIN_TYPE_DICT,
    ChainType,
    mock_sequence_id,
)
from boltz_finetune.utils.res_names import CCD_NAME_TO_ONE_LETTER
from boltz.data.types import MSA, Record, Target, StructureV2, Coords
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data import const
from boltz.data.parse.schema import get_template_records_from_search
from boltz_finetune.utils.hhsearch import HHSearch
from boltz.data.parse.mmcif import parse_mmcif
from rdkit.Chem.rdchem import Conformer, Mol
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv

# sequence_id mod_index ccd_code one_letter_code
ModifiedResidueId: TypeAlias = tuple[int, str, str]

# id chain_type sequence
ChainData: TypeAlias = tuple[str, ChainType, str]
BondAtomId: TypeAlias = tuple[str, int, str]

DEFAULT_API_SERVER: Final[str] = "https://api.colabfold.com"
COMMON_CRYSTALLIZATION_AIDS: Final[frozenset[str]] = frozenset(
    {
        "SO4",
        "GOL",
        "EDO",
        "PO4",
        "ACT",
        "PEG",
        "DMS",
        "TRS",
        "PGE",
        "PG4",
        "FMT",
        "EPE",
        "MPD",
        "MES",
        "CD",
        "IOD",
    }
)


class CyclicType(Enum):
    CyclicRight = 1
    CyclicError = 2
    CyclicFaild = 3


class ChiralityType(Enum):
    Chirality_L = 1
    Chirality_D = 2
    Chirality_None = 3


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    return False


def get_modified_residue(ccd_result: dict) -> tuple[bool, str]:
    is_modified = False
    target_residue = "X"
    acid_str: str = ccd_result["_chem_comp.name"][0].upper()
    sort_key = lambda x: len(x)
    sorted_key = sorted(list(ACID_NAME_2RES_DICT.keys()), key=sort_key, reverse=True)
    for key in sorted_key:
        if key in acid_str:
            is_modified = True
            target_residue = RES2ACID_DICT[ACID_NAME_2RES_DICT[key]]
            break

    if not is_modified and not ccd_result["_chem_comp.mon_nstd_parent_comp_id"][0] == "?":
        origin_residue = ccd_result["_chem_comp.mon_nstd_parent_comp_id"][0]
        if origin_residue in RES2ACID_DICT.keys():
            is_modified = True
            target_residue = RES2ACID_DICT[origin_residue]

    if not is_modified and not ccd_result["_chem_comp.one_letter_code"][0] == "?":
        is_modified = True
        target_residue = ccd_result["_chem_comp.one_letter_code"][0]

    return is_modified, target_residue


def parse_modified_sequence(squence_str: str) -> tuple[str, List[ModifiedResidueId]]:
    pattern = r"\((\w+)\)"
    modified_residues: List[ModifiedResidueId] = []
    matches = re.finditer(pattern, squence_str)
    parts = []
    ori_index = 0
    index = 0
    for match_patt in matches:

        parts.append(squence_str[ori_index : match_patt.start()])

        modified_residue = match_patt.group(1)
        ccd_result = None
        if ccd_result:

            is_modified, ori_residue = get_modified_residue(ccd_result)

            if is_modified:
                one_letter_code = ori_residue
            else:
                one_letter_code = modified_residue
        else:
            one_letter_code = "X"

        ori_index = match_patt.end()

        index += len(parts[-1]) + 1
        parts.append(one_letter_code)
        modified_residues.append((index, modified_residue, one_letter_code))
    parts.append(squence_str[ori_index:])
    return "".join(parts), modified_residues


def parse_fasta_lines(path: Path) -> List[str]:
    """Parses a fasta file and returns the sequence."""
    with open(path, "r") as f:
        lines = f.readlines()
    sequences: List[str] = []
    for line in lines:
        if line.startswith(">"):
            continue
        sequences.append(line.strip())
    return sequences


def parse_fasta_file(path: Path) -> tuple[List[ChainData], dict[int, List[ModifiedResidueId]]]:
    """Parses a fasta file and returns the sequence."""
    with open(path, "r") as f:
        lines = f.readlines()
    sequences: List[ChainData] = []
    modified_infos: dict[int, List[ModifiedResidueId]] = {}

    chain_id_str = ""
    chain_type = ChainType.Protein
    for i, line in enumerate(lines):
        if line.startswith(">"):
            line_lower = line.lower()
            if "protein" in line_lower:
                chain_type = ChainType.Protein
            if "dna" in line_lower:
                chain_type = ChainType.DNA
            if "rna" in line_lower:
                chain_type = ChainType.RNA
            if "ligand" in line_lower:
                chain_type = ChainType.Ligand
            chain_id_str = line.split("|")[1][-1]

            continue
        if "(" in line:
            sequence, modified_info = parse_modified_sequence(line.strip())
            modified_infos[i] = modified_info
            sequences.append((chain_id_str, chain_type, sequence))
            continue
        sequences.append((chain_id_str, chain_type, line.strip()))

    return sequences, modified_infos


def parse_fasta_extra_file(
    path: Path,
) -> tuple[List[tuple[BondAtomId, BondAtomId]], dict[int, List[ModifiedResidueId]]]:
    """Parses a fasta file and returns the sequence."""
    with open(path, "r") as f:
        lines = f.readlines()

    bond_pairs: List[tuple[BondAtomId, BondAtomId]] = []
    modified_infos: dict[int, List[ModifiedResidueId]] = {}
    bond_tag_end = 6
    modified_tag_end = 9
    for i, line in enumerate(lines):
        if "bonds" in line:
            bonds = line[bond_tag_end:].strip().split(",")
            for bond in bonds:
                bond_atoms = bond.split("-")
                assert len(bond_atoms) == 2, "bond atom must be two"
                bond_start = bond_atoms[0].split("_")
                bond_end = bond_atoms[1].split("_")
                assert len(bond_start) == 3 and len(bond_end) == 3, "bond atom must be three params"
                bond_pairs.append(
                    (
                        (bond_start[0], int(bond_start[1]), bond_start[2]),
                        (bond_end[0], int(bond_end[1]), bond_end[2]),
                    )
                )

        if "modified" in line:
            modified_str = line[modified_tag_end:].strip()
            if modified_str == "":
                continue
            modified_info = modified_str.split(",")
            for modified in modified_info:
                modified_residue = modified.split("_")
                assert len(modified_residue) == 4, "modified residue must be three params"
                index = int(modified_residue[0])
                if index not in modified_infos.keys():
                    modified_infos[index] = []
                modified_infos[index].append((int(modified_residue[1]), modified_residue[2], modified_residue[3]))

    return bond_pairs, modified_infos


def parse_pbd_file(path: str, with_insert: bool = False) -> tuple[List[ChainData], dict[int, List[ModifiedResidueId]]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", path)[0]
    chains = list(structure.get_chains())
    chains_num = len(chains)

    all_chain: dict[str, ChainData] = {}

    modified_infos: dict[int, List[ModifiedResidueId]] = {}

    ligand_num = 0
    insert_infos = {}
    for chain_index in range(chains_num):
        chain_type = ChainType.Protein
        res_list: List[Residue] = chains[chain_index].child_list
        chain_id = chains[chain_index].get_id()
        all_chain[chain_id] = ()
        raw_res_dict = {}

        for i, res in enumerate(res_list):
            name = res.get_resname()
            index = f"{res.get_id()[1]}{res.get_id()[2]}".strip()
            if name == "HOH":
                continue

            if len(name) >= 3:
                if name not in RES2ACID_DICT.keys():
                    if name not in COMMON_CRYSTALLIZATION_AIDS and not name == "HOH":

                        one_letter_code = name
                        ccd_result = ccd.get(name)
                        if name in CCD_NAME_TO_ONE_LETTER:

                            is_modified = True
                            ori_residue = CCD_NAME_TO_ONE_LETTER[name]

                            if is_modified:
                                one_letter_code = ori_residue

                            if len(ori_residue) >= 3 or not is_modified:
                                ligand_id = f"X_{ligand_num}"
                                all_chain[ligand_id] = (ligand_id, ChainType.Ligand, name)
                                ligand_num + 1

                                continue

                        else:
                            one_letter_code = "X"
                        if chain_index not in modified_infos.keys():
                            modified_infos[chain_index] = []

                        modified_infos[chain_index].append((index, name, one_letter_code))
                        raw_res_dict[index] = one_letter_code
                    else:
                        continue
                else:
                    raw_res_dict[index] = RES2ACID_DICT[name]
            if len(name) == 1:
                chain_type = ChainType.RNA
                if name in ["A", "G", "C", "U"]:
                    chain_type = ChainType.RNA

                if name == "T":
                    chain_type = ChainType.DNA

                raw_res_dict[index] = name

        seq = []
        insert_rela_dict = {}
        for res_key, res_str in raw_res_dict.items():
            seq.append(res_str)
            insert_rela_dict[res_key] = len(seq)
        all_chain[chain_id] = (chain_id, chain_type, "".join(seq))
        insert_infos[chain_index] = insert_rela_dict

    for chain_index, insert_rela_dict in insert_infos.items():
        if chain_index not in modified_infos:
            continue
        ptms_with_insert = []
        for ptm in modified_infos[chain_index]:
            index, name, one_letter_code = ptm
            if index in insert_rela_dict:
                ptms_with_insert.append((insert_rela_dict[index], name, one_letter_code))

        modified_infos[chain_index] = ptms_with_insert

    id_i = 0
    sequences_ids = list(all_chain.keys())
    sequences: List[ChainData] = []
    for seq_id, seq in all_chain.items():
        chain_id = seq_id
        if seq[1] == ChainType.Ligand:
            while mock_sequence_id(id_i) in sequences_ids:
                id_i += 1
            chain_id = mock_sequence_id(id_i)
            sequences_ids.append(chain_id)
        sequences.append((chain_id, seq[1], seq[-1]))
    if with_insert:
        return sequences, modified_infos, insert_infos

    return sequences, modified_infos


def mock_peptiede_boltz_data(
    sequences: List[ChainData],
    modified_info: dict[int, List[ModifiedResidueId]],
    transed_bond_pairs: List[tuple[BondAtomId, BondAtomId]],
    is_cyclic: bool = False,
) -> dict[str, Any]:
    result = {"version": 1}

    chains = []
    for i, chain in enumerate(sequences):
        chain_id, chain_type, sequence = chain
        ptms = []
        if i in modified_info.keys():
            for ptm in modified_info[i]:
                ptms.append({"position": int(ptm[0]), "ccd": ptm[1]})
        chain_type_tag = CHAIN_TYPE_DICT[chain_type]
        chain_dict = {}

        chain_data_dict = {}
        chain_data_dict["id"] = chain_id
        if ChainType.Ligand == chain_type:
            chain_data_dict["id"] = [chain_id]
            if sequence in CCD_NAME_TO_ONE_LETTER or len(sequence) < 4:
                chain_data_dict["ccd"] = sequence
            else:
                chain_data_dict["smiles"] = sequence
        else:
            chain_data_dict["id"] = chain_id
            chain_data_dict["sequence"] = sequence
            if is_cyclic:
                chain_data_dict["cyclic"] = True

        if len(ptms) > 0:
            chain_data_dict["modifications"] = ptms

        chain_dict[chain_type_tag] = chain_data_dict
        chains.append(chain_dict)

    result["sequences"] = chains

    if len(transed_bond_pairs) > 0:
        constraints = []
        for i, bond_pair in enumerate(transed_bond_pairs):
            constraints.append({"bond": {"atom1": list(bond_pair[0]), "atom2": list(bond_pair[1])}})
        result["constraints"] = constraints

    return result


def assign_atom_coords_from_pdb(pdb_file: Path, boltz_structure: StructureV2) -> dict[str, list[float]]:
    atom_coords = {}
    parser = PDBParser(QUIET=True) if pdb_file.suffix.lower() == ".pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)[0]
    res_index = 0
    for chain in structure:
        for residue in chain:
            res_dict = {}
            for atom in residue:
                res_dict[atom.name] = atom.get_coord()
            atom_coords[(res_index, residue.resname)] = res_dict
            res_index += 1

    res_num = len(boltz_structure.residues)
    for res_idx in range(res_num):
        cur_res = boltz_structure.residues[res_idx]
        cur_res_atom_start = cur_res["atom_idx"]
        atom_num = cur_res["atom_num"]
        for atom_idx in range(cur_res_atom_start, cur_res_atom_start + atom_num):
            atom_name = boltz_structure.atoms[atom_idx]["name"]
            if atom_name in atom_coords[(res_idx, cur_res["name"])] and atom_name in ["N", "CA", "C", "O", "CB"]:
                boltz_structure.atoms[atom_idx]["coords"] = atom_coords[(res_idx, cur_res["name"])][atom_name]
            else:
                # print(f"atom {atom_name} not found in {res_idx, cur_res['name']}")
                boltz_structure.atoms[atom_idx]["is_present"] = False
    coords = [(x,) for x in boltz_structure.atoms["coords"]]
    coords = np.array(coords, Coords)
    boltz_structure.coords = coords
    return boltz_structure


def assign_templates_from_search(
    to_generate: dict[str, str],
    boltz_target: Target,
    env_dir: Path = Path(""),
    ccd: dict[str, Mol] = {},
    mol_dir: Path = Path("/home/admin/.boltz/mols"),
):

    sequences = list(to_generate.keys())
    a3m_lines, template_path = run_mmseqs2(
        sequences,
        str(env_dir),
        use_env=True,
        use_pairing=False,
        host_url=DEFAULT_API_SERVER,
        pairing_strategy="greedy",
        use_templates=True,
    )
    if not all(template_path):
        return

    for i, a3m_line in enumerate(a3m_lines):

        if not a3m_line:
            continue
        hhsearch_pdb70_runner = HHSearch(binary_path="hhsearch", databases=[f"{template_path[i]}/pdb70"])

        hhsearch_result = hhsearch_pdb70_runner.query(a3m_line)

        hhsearch_hits = hhsearch_pdb70_runner.get_template_hits(
            hhsearch_result,
        )
        hhsearch_hits = sorted(hhsearch_hits, key=lambda x: x.sum_probs, reverse=True)
        search_hits = [x.name for x in hhsearch_hits if x.sum_probs >= 20][:MAZ_TEMPLATES_NUM]
        searched_files = []

        for hit in search_hits:
            template_cif, _ = tuple(hit.split("_"))
            file_path = Path(template_path[i]) / f"{template_cif}.cif"
            parsed_template = parse_mmcif(
                file_path,
                mols=ccd,
                moldir=mol_dir,
                use_assembly=False,
                compute_interfaces=False,
            )
            template_proteins = {
                str(c["name"]) for c in parsed_template.data.chains if c["mol_type"] == const.chain_type_ids["PROTEIN"]
            }
            template_chain_ids = list(template_proteins)
            matched_template = get_template_records_from_search(
                template_id=template_cif,
                chain_ids=[to_generate[sequences[i]]],
                sequences={to_generate[sequences[i]]: sequences[i]},
                template_chain_ids=template_chain_ids,
                template_sequences=parsed_template.sequences,
            )
            boltz_target.record.templates.extend(matched_template)
            boltz_target.templates[template_cif] = parsed_template.data


def handle_template(
    boltz_target: Target,
    use_template: bool = False,
    template_dict: dict[str, list[Path]] = {},
    env_dir: Path = Path(""),
    ccd: dict[str, Mol] = {},
    mol_dir: Path = Path("/home/admin/.boltz/mols"),
) -> None:
    if not use_template:
        return

    to_generate = {}
    prot_id = const.chain_type_ids["PROTEIN"]
    for chain in boltz_target.record.chains:
        # Add to generate list, assigning entity id
        if (chain.mol_type == prot_id) and (chain.msa_id == 0):
            entity_id = chain.entity_id
            to_generate[boltz_target.sequences[entity_id]] = chain.chain_name

    if not template_dict:
        assign_templates_from_search(
            to_generate,
            boltz_target,
            env_dir=env_dir,
            ccd=ccd,
            mol_dir=mol_dir,
        )
        return

    sequences = list(to_generate.keys())
    for i, sequence in enumerate(sequences):
        chain_id = to_generate[sequence]
        template_files = template_dict.get(chain_id, [])

        for template_file in template_files:
            parsed_template = parse_mmcif(
                template_file,
                mols=ccd,
                moldir=mol_dir,
                use_assembly=False,
                compute_interfaces=False,
            )
            template_id = template_file.stem
            template_proteins = {
                str(c["name"]) for c in parsed_template.data.chains if c["mol_type"] == const.chain_type_ids["PROTEIN"]
            }
            template_chain_ids = list(template_proteins)
            matched_template = get_template_records_from_search(
                template_id=template_id,
                chain_ids=[chain_id],
                sequences={chain_id: sequence},
                template_chain_ids=template_chain_ids,
                template_sequences=parsed_template.sequences,
            )
            boltz_target.record.templates.extend(matched_template)
            boltz_target.templates[template_id] = parsed_template.data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    use_mock: bool = False,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = []
    if use_mock:
        for seq in list(data.values()):
            unpaired_msa.append(f">101\n{seq}\n>101\n{seq}\n")
    else:
        unpaired_msa = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_unpaired_tmp",
            use_env=True,
            use_pairing=False,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_boltz_input(  # noqa: C901, PLR0912, PLR0915, D103
    target: Target,
    msa_dir: Path,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
    use_mock_msa: bool = False,
) -> Record | None:
    try:
        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                use_mock=use_mock_msa,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)
        return target.record

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error processing target {target.record.id}: {e}")
        return None
