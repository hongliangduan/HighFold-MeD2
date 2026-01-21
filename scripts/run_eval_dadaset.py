import os
from pathlib import Path
from time import time

from loguru import logger
from boltz.data.types import Manifest
from boltz_finetune.utils.infer_utils import get_data_module, get_pred_writer, get_model_module
from boltz_finetune.metrics import MetricsRunner
import pandas as pd

if __name__ == "__main__":
    logger.add(
        f"{Path(__file__).parent.parent}/log/eval_{str(time())}.log",
        format="{time} | {file} | {line} | {level} | {message}",
        level="INFO",
        colorize=False,
    )
    # test_data
    ## test processed data
    manifest_path = Path("/home/lab/dataset/processed/test.json")
    ## test data file
    data_set_dir = Path("/home/lab/dataset/test")

    # mol dir for ccd. Don't change it.
    mol_dir = Path("/home/lab/.boltz/mols/")

    # evalt file output
    data_tag = "test"
    out_dir = Path(f"/home/lab/test")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    # train model save path
    model_ckpt_path = Path(f"/home/lab/boltz_output/{data_tag}/checkpoints")

    # if only stats plz set need_pred to False
    need_pred = True

    manifest = Manifest.load(manifest_path)
    data_module = get_data_module(data_set_dir, manifest, mol_dir)

    ckpts = list(model_ckpt_path.glob("*.ckpt"))

    if need_pred:
        for ckpt in ckpts:
            ckpt = Path(ckpt)
            logger.info(f"Start Predict {ckpt}")
            model_out_dir = out_dir / ckpt.stem

            pred_writer = get_pred_writer(data_set_dir, model_out_dir)
            trainer = Trainer(
                default_root_dir=model_out_dir,
                callbacks=[pred_writer],
                accelerator="gpu",
                devices=[2, 3],
                precision="bf16-mixed",
                strategy="auto",
            )

            model_module = get_model_module(ckpt)
            trainer.predict(
                model_module,
                datamodule=data_module,
                return_predictions=False,
            )

    mean_dfs = []
    for ckpt in ckpts:
        ckpt = Path(ckpt)
        logger.info(f"Start Stat {ckpt}")
        model_out_dir = out_dir / ckpt.stem
        metrics_runner = MetricsRunner(use_multi_process=False)
        metrics_from_dir = metrics_runner.run_boltz_dataset_eval(data_set_dir, model_out_dir)
        df = pd.DataFrame(metrics_from_dir)
        index = df["id"]
        df = df[["ca_rmsd", "backbone_rmsd", "all_atom_rmsd", "uaa_atom_rmsd"]]
        mean_df = df.mean()
        df.loc["mean"] = mean_df
        df["id"] = index
        df.to_csv(model_out_dir / "metrics.csv")

        mean_dfs.append(
            {
                "Model": ckpt.stem,
                "RMSD-ALL": mean_df["all_atom_rmsd"],
                "RMSD-CA": mean_df["ca_rmsd"],
                "RMSD-Backbone": mean_df["backbone_rmsd"],
                "RMSD-UAA": mean_df["uaa_atom_rmsd"],
            }
        )
        logger.info(
            f"\nRMSD-ALL: {mean_df['all_atom_rmsd']:.4f} \nRMSD-CA: {mean_df['ca_rmsd']:.4f}\nRMSD-Backbone: {mean_df['backbone_rmsd']:.4f}\nRMSD-UAA: {mean_df['uaa_atom_rmsd']:.4f}\n"
        )

    result = pd.DataFrame(mean_dfs)
    logger.info(result)
    result.to_csv(f"{out_dir}/eval_result.csv", index=False)
