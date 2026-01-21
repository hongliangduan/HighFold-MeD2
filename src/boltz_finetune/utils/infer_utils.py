import json
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Input,
    Interface,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
    StructureV2,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from torch.utils.data import DataLoader
from boltz.model.models.boltz2 import Boltz2


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = True
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    guidance_update: bool = True
    num_gd_steps: int = 20


def load_input(
    record: Record,
    data_set_dir: Path,
    extra_mols_dir: Optional[Path] = None,
    affinity: bool = False,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.
    constraints_dir : Optional[Path]
        The path to the constraints directory.
    template_dir : Optional[Path]
        The path to the template directory.
    extra_mols_dir : Optional[Path]
        The path to the extra molecules directory.
    affinity : bool
        Whether to load the affinity data.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure

    # if affinity:
    #     structure = StructureV2.load(target_dir / record.id / f"pre_affinity_{record.id}.npz")
    # else:
    process_dir = data_set_dir / record.id / "boltz_data" / "processed"

    structure = StructureV2.load(process_dir / f"structures/{record.id}.npz")

    msas = {}
    msa_dir = process_dir / "msa"
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = MSA.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = msa

    # Load templates
    templates = None
    if record.templates:
        templates = {}
        for template_info in record.templates:
            template_id = template_info.name
            template_path = process_dir / "templates" / f"{record.id}_{template_id}.npz"
            template = StructureV2.load(template_path)
            templates[template_id] = template

    # Load residue constraints
    residue_constraints = ResidueConstraints.load(process_dir / "constraints" / f"{record.id}.npz")

    # Load extra molecules
    extra_mols = {}
    if extra_mols_dir is not None:
        extra_mol_path = process_dir / "mols" / f"{record.id}.pkl"
        if extra_mol_path.exists():
            with extra_mol_path.open("rb") as f:
                extra_mols = pickle.load(f)  # noqa: S301

    return Input(
        structure,
        msas,
        record=record,
        residue_constraints=residue_constraints,
        templates=templates,
        extra_mols=extra_mols,
    )


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        data_set_dir: Path,
        mol_dir: Path,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.

        """
        super().__init__()
        self.manifest = manifest
        self.data_set_dir = data_set_dir
        self.mol_dir = mol_dir

        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(self.mol_dir)
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        if self.affinity:
            self.cropper = AffinityCropper()

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get record
        record = self.manifest.records[idx]

        # Finalize input data
        input_data = load_input(
            record=record,
            data_set_dir=self.data_set_dir,
            extra_mols_dir=self.extra_mols_dir,
            affinity=self.affinity,
        )

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        if self.affinity:
            try:
                tokenized = self.cropper.crop(
                    tokenized,
                    max_tokens=256,
                    max_atoms=2048,
                )
            except Exception as e:  # noqa: BLE001
                print(f"Cropper failed on {record.id} with error {e}. Skipping.")  # noqa: T201
                return self.__getitem__(0)

        # Load conformers
        try:
            molecules = {}
            molecules.update(self.canonicals)
            molecules.update(input_data.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.mol_dir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            pocket_constraints = None, None
        else:
            pocket_constraints = options.pocket_constraints

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                override_method=self.override_method,
                compute_affinity=self.affinity,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class Boltz2InferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz2 inference."""

    def __init__(
        self,
        manifest: Manifest,
        data_set_dir: Path,
        mol_dir: Path,
        num_workers: int,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        mol_dir : Path
            The path to the moldir.
        num_workers : int
            The number of workers to use.
        constraints_dir : Optional[Path]
            The path to the constraints directory.
        template_dir : Optional[Path]
            The path to the template directory.
        extra_mols_dir : Optional[Path]
            The path to the extra molecules directory.
        override_method : Optional[str]
            The method to override.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.data_set_dir = data_set_dir
        self.mol_dir = mol_dir
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            data_set_dir=self.data_set_dir,
            mol_dir=self.mol_dir,
            extra_mols_dir=self.extra_mols_dir,
            override_method=self.override_method,
            affinity=self.affinity,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch


class BoltzBatchWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records

        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / record.id / "boltz_data" / "processed" / "structures" / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2))
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2))
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = struct_dir / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        if key not in prediction:
                            continue
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    if "pair_chains_iptm" in prediction:
                        confidence_summary_dict["chains_ptm"] = {
                            idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                            for idx in prediction["pair_chains_iptm"]
                        }
                        confidence_summary_dict["pair_chains_iptm"] = {
                            idx1: {
                                idx2: prediction["pair_chains_iptm"][idx1][idx2][model_idx].item()
                                for idx2 in prediction["pair_chains_iptm"][idx1]
                            }
                            for idx1 in prediction["pair_chains_iptm"]
                        }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = struct_dir / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = struct_dir / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = struct_dir / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


def get_data_module(data_set_dir: Path, manifest: Manifest, mol_dir: Path):
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        data_set_dir=data_set_dir,
        mol_dir=mol_dir,
        num_workers=2,
    )
    return data_module


def get_pred_writer(data_dir: Path, out_dir: Path):

    # Create prediction writer
    pred_writer = BoltzBatchWriter(
        data_dir=data_dir,
        output_dir=out_dir,
        output_format="pdb",
        boltz2=True,
    )
    return pred_writer


def get_model_module(checkpoint: Path, step_scale: float = None):

    diffusion_params = Boltz2DiffusionParams()
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params.step_scale = step_scale
    pairformer_args = PairformerArgsV2()

    msa_args = MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )

    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 5,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.guidance_update = False

    model_module = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()
    return model_module
