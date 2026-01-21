from pathlib import Path

from boltz_finetune.common.config import (
    DATA_OUTPUT_DIR_NAME,
    DATA_OUTPUT_SUFFIX,
    EXTRA_FILE_SUFFIX,
    SEQUENCE_FILE_SUFFIX,
    STRUCTURE_FILE_SUFFIX,
    TEMPLATE_FILE_SUFFIX,
    USE_MSA,
    USE_TEMPLATE,
)
from boltz_finetune.utils.data_utils import load_yaml_as_dict
from boltz.data.mol import load_canonicals


class ProcessConfig:
    def __init__(
        self,
        data_root_dir: str,
        sequence_file_suffix: str = SEQUENCE_FILE_SUFFIX,
        structure_file_suffix: str = STRUCTURE_FILE_SUFFIX,
        extra_file_suffix: str = EXTRA_FILE_SUFFIX,
        data_output_suffix: str = DATA_OUTPUT_SUFFIX,
        data_output_dir_name: str = DATA_OUTPUT_DIR_NAME,
        records_file_suffix: str = "json",
        use_msa: bool = USE_MSA,
        use_template: bool = USE_TEMPLATE,
        is_cyclic: bool = False,
        template_file_suffix: str = TEMPLATE_FILE_SUFFIX,
        user_template_path: str = None,
        mol_dir: str = "/home/admin/.boltz/mols",
        msa_dir_name: str = "msa",
        records_dir_name: str = "records",
        structure_dir_name: str = "structures",
        constraints_dir_name: str = "constraints",
        templates_dir_name: str = "templates",
        mols_dir_name: str = "mols",
    ):

        self.data_root_dir = Path(data_root_dir)
        self.sequence_file_suffix = sequence_file_suffix
        self.structure_file_suffix = structure_file_suffix
        self.extra_file_suffix = extra_file_suffix
        self.records_file_suffix = records_file_suffix

        self.use_msa = use_msa
        self.use_template = use_template
        self.is_cyclic = is_cyclic
        self.template_file_suffix = template_file_suffix
        self.user_template_path = Path(user_template_path)

        self.data_output_dir_name = data_output_dir_name
        self.data_output_suffix = data_output_suffix
        self.train_data_file = self.data_root_dir / f"{data_output_dir_name}/train.{data_output_suffix}"
        self.validation_data_file = self.data_root_dir / f"{data_output_dir_name}/valid.{data_output_suffix}"
        self.test_data_file = self.data_root_dir / f"{data_output_dir_name}/test.{data_output_suffix}"

        self.mol_dir = Path(mol_dir)
        self.ccd = load_canonicals(self.mol_dir)

        self.msa_dir_name = msa_dir_name
        self.records_dir_name = records_dir_name
        self.structure_dir_name = structure_dir_name
        self.constraints_dir_name = constraints_dir_name
        self.templates_dir_name = templates_dir_name
        self.mols_dir_name = mols_dir_name

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "ProcessConfig":
        yaml_config = load_yaml_as_dict(yaml_file)
        return cls(
            yaml_config["data_root_dir"],
            yaml_config["sequence_file_suffix"],
            yaml_config["structure_file_suffix"],
            yaml_config["extra_file_suffix"],
            yaml_config["data_output_suffix"],
            yaml_config["data_output_dir_name"],
            yaml_config["records_file_suffix"],
            yaml_config["use_msa"],
            yaml_config["use_template"],
            yaml_config["is_cyclic"],
            yaml_config["template_file_suffix"],
            yaml_config["user_template_path"],
            yaml_config["mol_dir"],
            yaml_config["msa_dir_name"],
            yaml_config["records_dir_name"],
            yaml_config["structure_dir_name"],
            yaml_config["constraints_dir_name"],
            yaml_config["templates_dir_name"],
            yaml_config["mols_dir_name"],
        )
