# load and generate train data set .
# process fasta file and pdb file.

import shutil
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from boltz.data.types import Manifest, Record
from boltz_finetune.process.process_config import ProcessConfig
from boltz_finetune.utils.data_utils import dump_data, get_dir_files, load_data
from boltz_finetune.utils.input_utils import (
    check_modified_residues,
    check_sequences,
    parse_fasta_extra_file,
    parse_fasta_file,
    parse_pbd_file,
    trans_bond_pairs,
    mock_peptiede_boltz_data,
    handle_template,
    process_boltz_input,
    DEFAULT_API_SERVER,
    assign_atom_coords_from_pdb,
)
from boltz.data.parse.schema import parse_boltz_schema


@dataclass
class DataProcess:
    config: ProcessConfig

    def __post_init__(self):
        self.dataset = []

    def process_train_input(
        self,
        sequence_file: Path,
        structure_file: Path,
        extra_file: Path,
        template_Path: Path,
        out_dir: Path,
        is_trainning: bool = True,
        *,
        check_data: bool = False,
        use_msa: bool = True,
        use_template: bool = False,
    ) -> Manifest:
        task_id = sequence_file.stem.upper()
        sequences, modified_info = parse_fasta_file(sequence_file)

        pdb_file = sequence_file.with_suffix(".cif")
        if pdb_file.exists() and check_data:
            pdb_sequences, pdb_modified_infos = parse_pbd_file(pdb_file)
            assert check_sequences(sequences, pdb_sequences), "fasta and pdb sequences are not same"
            assert check_modified_residues(
                modified_info, pdb_modified_infos
            ), "fasta and pdb modified residues are not same"
            if len(modified_info) < 1:
                modified_info.update(pdb_modified_infos)

        bond_pairs = []
        if extra_file.exists():
            bond_pairs, extra_modified_infos = parse_fasta_extra_file(extra_file)
            assert check_modified_residues(
                modified_info, extra_modified_infos
            ), "fasta and extra modified residues are not same"
            modified_info.update(extra_modified_infos)

        transed_bond_pairs = trans_bond_pairs(bond_pairs, sequences)
        boltz_data = mock_peptiede_boltz_data(
            sequences, modified_info, transed_bond_pairs, is_cyclic=self.config.is_cyclic
        )

        boltz_target = parse_boltz_schema(task_id, boltz_data, self.config.ccd, self.config.mol_dir, boltz_2=True)
        if is_trainning:
            assign_atom_coords_from_pdb(pdb_file, boltz_target.structure)

        env_dir = out_dir / "envs"
        fixed_template_path = (
            template_Path / f"{sequence_file.stem.upper()}_template.{self.config.template_file_suffix}"
        )
        template_dict = None
        if fixed_template_path.exists():
            template_dict = {"A": [fixed_template_path]}
        else:
            logger.warning(f"No template found for {fixed_template_path}")

        handle_template(
            boltz_target,
            template_dict=template_dict,
            use_template=use_template,
            env_dir=env_dir,
            mol_dir=self.config.mol_dir,
        )

        msa_dir = out_dir / self.config.msa_dir_name
        records_dir = out_dir / self.config.data_output_dir_name / self.config.records_dir_name
        structure_dir = out_dir / self.config.data_output_dir_name / self.config.structure_dir_name
        processed_msa_dir = out_dir / self.config.data_output_dir_name / self.config.msa_dir_name
        processed_constraints_dir = out_dir / self.config.data_output_dir_name / self.config.constraints_dir_name
        processed_templates_dir = out_dir / self.config.data_output_dir_name / self.config.templates_dir_name
        processed_mols_dir = out_dir / self.config.data_output_dir_name / self.config.mols_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        msa_dir.mkdir(parents=True, exist_ok=True)
        records_dir.mkdir(parents=True, exist_ok=True)
        structure_dir.mkdir(parents=True, exist_ok=True)
        processed_msa_dir.mkdir(parents=True, exist_ok=True)
        processed_constraints_dir.mkdir(parents=True, exist_ok=True)
        processed_templates_dir.mkdir(parents=True, exist_ok=True)
        processed_mols_dir.mkdir(parents=True, exist_ok=True)

        return process_boltz_input(
            boltz_target,
            msa_dir=msa_dir,
            use_msa_server=True,
            msa_server_url=DEFAULT_API_SERVER,
            msa_pairing_strategy="greedy",
            max_msa_seqs=8192,
            processed_msa_dir=processed_msa_dir,
            processed_constraints_dir=processed_constraints_dir,
            processed_templates_dir=processed_templates_dir,
            processed_mols_dir=processed_mols_dir,
            structure_dir=structure_dir,
            records_dir=records_dir,
            use_mock_msa=not use_msa,  # use_mock_msa=True,
        )

    def process_single_data_file(self, sequnce_file: Path, data_tag: str, reprocess: bool = False) -> None:

        structure_file = sequnce_file.with_suffix(f".{self.config.template_file_suffix}")
        extra_file = sequnce_file.with_suffix(f".{self.config.extra_file_suffix}")
        out_dir = structure_file.parent / "boltz_data"
        record_file = (
            out_dir
            / self.config.data_output_dir_name
            / self.config.records_dir_name
            / (sequnce_file.stem.upper() + ".json")
        )
        if record_file.exists():
            if not reprocess:
                logger.warning(f"Processed file {record_file} already exists. Skipping.")
                record = Record.load(record_file)
                self.dataset.append(record)
                return
            else:
                shutil.rmtree(out_dir)

        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        record = self.process_train_input(
            sequnce_file,
            structure_file,
            extra_file,
            self.config.user_template_path,
            out_dir,
            is_trainning=not (data_tag == "test"),
            check_data=False,
            use_msa=self.config.use_msa,
            use_template=self.config.use_template,
        )
        if record is None:
            logger.warning(f"Processed file {sequnce_file} is None. Skipping.")
            return
        self.dataset.append(record)
        logger.info(f"Processed {data_tag} data file {sequnce_file} to {record_file}")

    def process_datasets(self, data_tag: str, save_catch: bool = True, reprocess: bool = False) -> None:
        sequence_files = get_dir_files(self.config.data_root_dir / data_tag, self.config.sequence_file_suffix)

        temp_processed_file = self.config.data_root_dir / self.config.data_output_dir_name / f"{data_tag}/"

        for sequnce_file in sequence_files:
            try:
                logger.info(f"Start process {str(sequnce_file)}.")
                self.process_single_data_file(sequnce_file, data_tag, reprocess)
            except Exception as e:
                logger.error(f"Error processing {str(sequnce_file)}: {str(e)}")

        if len(self.dataset) > 0:
            logger.info(f"Processed {len(self.dataset)} {data_tag} data files.")
            if not save_catch:
                logger.info(f"Not save catch data. Will remove {temp_processed_file}")
                shutil.rmtree(temp_processed_file)

        if data_tag == "train":
            manifest = Manifest(self.dataset)
            dump_data(manifest, self.config.train_data_file)

        if data_tag == "valid":
            manifest = Manifest(self.dataset)
            dump_data(manifest, self.config.validation_data_file)

        if data_tag == "test":
            manifest = Manifest(self.dataset)
            dump_data(manifest, self.config.test_data_file)

        self.dataset.clear()

    def load_datasets(self, data_tag: str) -> Manifest:

        if data_tag == "train":
            return load_data(self.config.train_data_file)

        if data_tag == "valid":
            return load_data(self.config.validation_data_file)

        if data_tag == "test":
            return load_data(self.config.test_data_file)
