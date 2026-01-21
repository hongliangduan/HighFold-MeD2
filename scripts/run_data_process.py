from pathlib import Path
from time import time

from loguru import logger

from boltz_finetune.process.data_process import DataProcess
from boltz_finetune.process.process_config import ProcessConfig

if __name__ == "__main__":
    logger.add(
        f"{Path(__file__).parent.parent}/log/data_process_{str(time())}.log",
        format="{time} | {file} | {line} | {level} | {message}",
        level="INFO",
        colorize=False,
    )

    yaml_file = Path("../configs/data_config.yaml")
    process_config = ProcessConfig.from_yaml(yaml_file)

    data_process = DataProcess(process_config)

    data_process.process_datasets("train", reprocess=True)

    manifest = data_process.load_datasets("train")
    manifest.dump(process_config.train_data_file.with_suffix(".json"))
    print(f"Loaded {len(manifest.records)} train records.")

    data_process.process_datasets("test", reprocess=True)
    test_manifest = data_process.load_datasets("test")
    test_manifest.dump(process_config.test_data_file.with_suffix(".json"))
    print(f"Loaded {len(test_manifest.records)} test records.")
