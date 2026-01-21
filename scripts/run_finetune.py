import os
from pathlib import Path

from loguru import logger

from boltz_finetune.finetuner import Finetuner

if __name__ == "__main__":

    yaml_file = Path("../configs/boltz2_fintune.yaml")
    finetuner = Finetuner(yaml_file, ["debug"])
    finetuner.run()
