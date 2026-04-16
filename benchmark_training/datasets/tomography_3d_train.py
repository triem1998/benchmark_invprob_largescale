"""Benchopt training dataset for 3D Walnut cone-beam CT.

Uses WalnutTomoDataset (CSV + np.memmap, lazy per-sample loading) with a
standard torch DataLoader and WalnutGroupSampler for reproducible group-level
shuffling.  Each batch is a dict with keys: ``"x"``, ``"x_sparse"``,
``"y_full"``, ``"num_proj"``, ``"sample_id"``, ``"operator_norm"``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from torch.utils.data import DataLoader

from toolsbench.utils import (
    load_trajectory_sparse,
    WalnutGroupSampler,
    WalnutTomoDataset,
)


class Dataset(BaseDataset):
    """3D Walnut CT training dataset using standard torch DataLoader.

    Wraps WalnutTomoDataset (lazy CSV + np.memmap loading) with WalnutGroupSampler
    for reproducible group-level shuffling.  Returns dict-format batches so the
    solver's _Trainer can track per-sample metadata (sample_id, operator_norm).
    """

    name = "tomography_3d_train"

    requirements = [
        "torch",
        "numpy",
        "pandas",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    parameters = {
        "input_dir": ["/lustre/fswork/projects/rech/fio/commun/Walnut-CBCT"],
        "num_projs": [[30, 50, 100]],
        "batch_size": [1],
        "max_train_samples": [None],
        "max_val_samples": [None],
        # None → num_operators == num_proj (one sub-operator per projection angle)
        "num_operators": [None],
        "num_workers": [2],
        "pin_memory": [True],
        "prefetch_factor": [2],
        "persistent_workers": [True],
    }

    def __init__(
        self,
        input_dir="/lustre/fswork/projects/rech/fio/commun/Walnut-CBCT",
        num_projs=(30, 50, 100),
        batch_size=1,
        max_train_samples=None,
        max_val_samples=None,
        num_operators=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    ):
        self.input_dir = Path(input_dir)
        self.num_projs = list(num_projs)
        self.batch_size = int(batch_size)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.num_operators = num_operators
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

    # ------------------------------------------------------------------
    # Physics factory
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Physics factory
    # ------------------------------------------------------------------

    def _make_physics_factory(self, trajectory_sparse, num_ops: int, img_size: tuple):
        """Angle-split physics factory (one TomographyWithAstra per distributed rank).

        The factory signature is ``factory(index, device, shared=None)`` as
        required by ``deepinv.distributed.distribute()``.
        """
        from toolsbench.utils import projection_splits

        splits = projection_splits(len(trajectory_sparse), num_ops)
        traj_cpu = trajectory_sparse.cpu()

        def factory(index: int, device, shared=None):
            import deepinv as dinv

            start, end = splits[index]
            traj_subset = traj_cpu[start:end].clone().to(device)
            print(
                f"[tomography_3d] physics factory op {index}: angles [{start}:{end}]",
                flush=True,
            )
            return dinv.physics.TomographyWithAstra(
                img_size=img_size,
                num_angles=end - start,
                num_detectors=(972, 768),
                pixel_spacing=(0.1, 0.1, 0.1),
                geometry_type="conebeam",
                geometry_vectors=traj_subset,
                normalize=False,
                device=device,
            )

        return factory

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_data(self):
        """Build train/val DataLoaders and per-config physics factories.

        Returns a dict with keys expected by ``Objective.set_data()``:
          ``train_dataloader``, ``val_dataloader``,
          ``physics`` (dict keyed by num_proj), ``num_operators`` (dict),
          ``ground_truth_shape``, ``min_pixel``, ``max_pixel``,
          ``operator_norm`` (mean scalar over all training samples),
          ``operator_norm_map`` (full per-sample dict).
        """
        # Export distributed env from submitit if in a SLURM job.
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            try:
                import submitit

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
                print("[tomography_3d] distributed env exported via submitit")
            except (ImportError, RuntimeError) as e:
                print(f"[tomography_3d] non-distributed mode ({e})")

        # Build datasets (lazy: no binary data loaded yet).
        train_dataset = WalnutTomoDataset(
            self.input_dir,
            self.num_projs,
            "train",
            max_samples=self.max_train_samples,
        )
        val_dataset = WalnutTomoDataset(
            self.input_dir,
            self.num_projs,
            "validation",
            max_samples=self.max_val_samples,
        )
        print(
            f"[tomography_3d] train_samples={len(train_dataset)} "
            f"val_samples={len(val_dataset)} num_projs={self.num_projs}",
            flush=True,
        )

        # Samplers with group-level shuffling.
        train_sampler = WalnutGroupSampler(train_dataset, shuffle=True, seed=0)
        val_sampler = WalnutGroupSampler(val_dataset, shuffle=False, seed=0)

        # DataLoader kwargs.
        lkw: dict = dict(
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        if self.num_workers > 0:
            lkw["prefetch_factor"] = self.prefetch_factor
            lkw["persistent_workers"] = self.persistent_workers

        train_loader = DataLoader(train_dataset, sampler=train_sampler, **lkw)
        val_loader = DataLoader(
            val_dataset, sampler=val_sampler, **{**lkw, "batch_size": 1}
        )

        # img_size from CSV metadata (no binary read needed).
        first_entry = train_dataset.entries[0]
        img_size = (
            first_entry.number_of_slice,
            first_entry.num_voxels,
            first_entry.num_voxels,
        )
        ground_truth_shape = (1, 1) + img_size  # (B=1, C=1, D, H, W)
        first_train_id = first_entry.sample_id

        # Per-config physics factories.
        physics_dict: dict = {}
        num_operators_dict: dict = {}
        for num_proj in self.num_projs:
            num_ops = (
                int(self.num_operators)
                if self.num_operators is not None
                else int(num_proj)
            )
            traj = load_trajectory_sparse(self.input_dir, first_train_id, num_proj)
            physics_dict[num_proj] = self._make_physics_factory(traj, num_ops, img_size)
            num_operators_dict[num_proj] = num_ops
            print(
                f"[tomography_3d] built physics factory num_proj={num_proj} "
                f"num_operators={num_ops}",
                flush=True,
            )

        # Pixel range from first sample (one np.memmap read).
        sample0 = train_dataset[0]
        x0 = sample0["x"]
        min_pixel = float(x0.min().item())
        max_pixel = float(x0.max().item())

        # Operator norms (loaded from opnorms.csv inside WalnutTomoDataset).
        norm_map = train_dataset.operator_norm_map  # {(sample_id, num_proj): float}
        operator_norm = float(np.mean(list(norm_map.values()))) if norm_map else 1.0
        if norm_map:
            norms = list(norm_map.values())
            print(
                f"[tomography_3d] operator_norm: mean={operator_norm:.4f} "
                f"min={min(norms):.4f} max={max(norms):.4f}",
                flush=True,
            )
        else:
            print("[tomography_3d] opnorms.csv not found — using 1.0 for all")

        print(
            f"[tomography_3d] done. shape={ground_truth_shape} "
            f"pixel=[{min_pixel:.3f},{max_pixel:.3f}]",
            flush=True,
        )

        return dict(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            physics=physics_dict,
            num_operators=num_operators_dict,
            ground_truth_shape=ground_truth_shape,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
            operator_norm=operator_norm,
            operator_norm_map=norm_map,
        )
