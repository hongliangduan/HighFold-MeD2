import os
import random
import string
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, listconfig
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.trainingv2 import BoltzTrainingDataModule, DataConfig
from boltz_finetune.utils.train_utils import TrainConfig


class Finetuner:
    def __init__(self, config_path: Path, args: list[str]):

        raw_config = omegaconf.OmegaConf.load(config_path)

        # Apply input arguments
        args = omegaconf.OmegaConf.from_dotlist(args)
        raw_config = omegaconf.OmegaConf.merge(raw_config, args)

        self.raw_config = raw_config
        # Instantiate the task
        cfg = hydra.utils.instantiate(raw_config)
        cfg = TrainConfig(**cfg)
        pl.seed_everything(cfg.data["random_seed"], workers=True)
        # Set matmul precision
        # if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision("high")

        # Create trainer dict
        self.trainer = cfg.trainer
        if self.trainer is None:
            self.trainer = {}

        devices = self.trainer.get("devices", 1)

        wandb = cfg.wandb
        if cfg.debug:
            if isinstance(devices, int):
                devices = 1
            elif isinstance(devices, (list, listconfig.ListConfig)):
                devices = [devices[0]]
            self.trainer["devices"] = devices
            cfg.data.num_workers = 0
            if wandb:
                wandb = None
        # init model
        model_module = cfg.model
        if cfg.pretrained and not cfg.resume:
            # Load the pretrained weights into the confidence module
            if cfg.load_confidence_from_trunk:
                checkpoint = torch.load(cfg.pretrained, map_location="cpu")

                # Modify parameter names in the state_dict
                new_state_dict = {}
                for key, value in checkpoint["state_dict"].items():
                    if not key.startswith("structure_module") and not key.startswith("distogram_module"):
                        new_key = "confidence_module." + key
                        new_state_dict[new_key] = value
                new_state_dict.update(checkpoint["state_dict"])

                # Update the checkpoint with the new state_dict
                checkpoint["state_dict"] = new_state_dict

                # Save the modified checkpoint
                random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
                file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
                print(
                    f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module."
                )
                torch.save(checkpoint, file_path)
            else:
                file_path = cfg.pretrained

            print(f"Loading model from {file_path}")
            model_module = type(model_module).load_from_checkpoint(
                file_path, map_location="cpu", strict=False, **(model_module.hparams)
            )

            if cfg.load_confidence_from_trunk:
                os.remove(file_path)

        callbacks = []

        if not cfg.disable_checkpoint:
            mc = ModelCheckpoint(
                monitor="val/rmsd",
                save_top_k=cfg.save_top_k,
                save_last=True,
                mode="min",
                every_n_epochs=1,
            )
            callbacks = [mc]

        self.callbacks = callbacks
        self.dirpath = cfg.output
        self.cfg = cfg
        self.devices = devices
        data_config = DataConfig(**cfg.data)
        self.data_module = BoltzTrainingDataModule(data_config)
        self.model_module = model_module
        self.wandb = wandb
        self.loggers = []
        # Save the config to wandb
        if wandb:
            self._save_config()

    def _save_config(self):

        wdb_logger = WandbLogger(
            name=self.wandb["name"],
            group=self.wandb["name"],
            save_dir=self.cfg.output,
            project=self.wandb["project"],
            entity=self.wandb["entity"],
            log_model=False,
        )
        self.loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(self.raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    def run(self):
        strategy = "auto"
        if (isinstance(self.devices, int) and self.devices > 1) or (
            isinstance(self.devices, (list, listconfig.ListConfig)) and len(self.devices) > 1
        ):
            strategy = DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
        # strategy = DeepSpeedStrategy(
        #     stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        #
        strategy = "ddp_find_unused_parameters_true"
        trainer = pl.Trainer(
            default_root_dir=str(self.dirpath),
            strategy=strategy,
            callbacks=self.callbacks,
            logger=self.loggers,
            enable_checkpointing=not self.cfg.disable_checkpoint,
            reload_dataloaders_every_n_epochs=1,
            **self.trainer,
        )

        if not self.cfg.strict_loading:
            self.model_module.strict_loading = False

        if self.cfg.validation_only:
            trainer.validate(
                self.model_module,
                datamodule=self.data_module,
                ckpt_path=self.cfg.resume,
            )
        else:
            trainer.fit(
                self.model_module,
                datamodule=self.data_module,
                ckpt_path=self.cfg.resume,
            )
