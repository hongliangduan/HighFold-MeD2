from pathlib import Path
from time import time

from loguru import logger

from boltz_finetune.process.data_process import DataProcess
from boltz_finetune.process.process_config import ProcessConfig

if __name__ == "__main__":

    yaml_file = Path("/home/fuxin/lab/wwt/boltz_finetune/configs/config.yaml")
    process_config = ProcessConfig.from_yaml(yaml_file)

    data_process = DataProcess(process_config)
    data_process.process_single_data_file(
        Path("/home/fuxin/lab/wwt/czg_datasets_v5/test/ME_7813AAASRESULT_PROC0081_0085/Me_7813AAAsresult_proc0081_0085.fasta"), data_tag="test", reprocess=True
    )
    print(data_process.dataset)
