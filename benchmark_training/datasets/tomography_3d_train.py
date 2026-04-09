"""Benchopt training dataset for 3D Walnut cone-beam CT, multi-config distributed.

For every walnut ID all sparsity configs are loaded in order:
    (id1, 30p), (id1, 50p), (id1, 100p), (id2, 30p), ...
giving ``n_ids × len(num_projs)`` samples (e.g. 27 × 3 = 81 by default).

The train loader shuffles at the group level (whole walnut-ID triplets stay
together) so each ``grad_accumulation_steps`` window always covers all
sparsity levels for the same walnut ID.

All tensors are stored on CPU; the solver moves each batch to GPU on the fly.
Each dataloader sample is a 4-tuple: ``(x, x_sparse, y_tensorlist, num_proj)``.
"""

from __future__ import annotations

import math
import os
import random as pyrandom
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from benchopt import BaseDataset
from deepinv.distributed import DistributedContext
from deepinv.utils.tensorlist import TensorList
from toolsbench.utils.walnut import WalnutMemoryMaps

# ---------------------------------------------------------------------------
# Minimal in-memory dataloader (no multiprocessing, safe for distributed use)
# ---------------------------------------------------------------------------


class ParallelReadDataLoader:
    """Simple iterable over pre-loaded (x, x_sparse, y_tensorlist, num_proj) tuples.

    Replaces a multiprocessing DataLoader (which caused hangs in distributed mode).

    When shuffle=True (train loader), the sample order is randomised on every
    call to __iter__ so consecutive optimizer steps see IID (walnut_id, num_proj)
    pairs drawn from the full training pool.

    When batch_size > 1, consecutive samples sharing the same num_proj are
    stacked along dim 0.  Batching across different num_proj values is not
    supported (y_tl angle dimensions are incompatible); a batch boundary is
    inserted whenever num_proj changes in the iteration order.
    """

    def __init__(
        self,
        samples: List,
        shuffle: bool = False,
        batch_size: int = 1,
        group_size: int = 1,
    ):
        self.samples = samples
        self.shuffle = shuffle
        self.batch_size = batch_size
        # When group_size > 1, shuffle preserves consecutive groups of this size
        # so that grad-accumulation windows always span all num_proj configs.
        self.group_size = group_size
        self.drop_last = False
        self.dataset = _DummyDataset(len(samples))

    @staticmethod
    def _stack_batch(chunk):
        """Stack a list of (x, x_sparse, y_tl, num_proj) tuples into one batch."""
        x = torch.cat([s[0] for s in chunk], dim=0)
        x_sparse = torch.cat([s[1] for s in chunk], dim=0)
        y_tl = TensorList(
            [
                torch.cat([s[2][k] for s in chunk], dim=0)
                for k in range(len(chunk[0][2]))
            ]
        )
        return (x, x_sparse, y_tl, chunk[0][3])

    def __iter__(self):
        if self.shuffle and self.group_size > 1:
            # Shuffle whole groups (triplets of 30p/50p/100p) together so that
            # consecutive grad-accumulation steps always cover all num_proj configs.
            n_groups = math.ceil(len(self.samples) / self.group_size)
            group_order = list(range(n_groups))
            pyrandom.shuffle(group_order)
            order = []
            for g in group_order:
                start = g * self.group_size
                end = min(start + self.group_size, len(self.samples))
                group_idxs = list(range(start, end))
                pyrandom.shuffle(group_idxs)  # shuffle config order within each triplet
                order.extend(group_idxs)
        else:
            order = list(range(len(self.samples)))
            if self.shuffle:
                pyrandom.shuffle(order)
        samples = [self.samples[i] for i in order]

        if self.batch_size == 1:
            return iter(samples)

        # Group consecutive same-num_proj samples into batches of batch_size.
        batched = []
        i = 0
        while i < len(samples):
            ref_key = samples[i][3]  # num_proj of this batch
            j = i + 1
            while (
                j < len(samples)
                and (j - i) < self.batch_size
                and samples[j][3] == ref_key
            ):
                j += 1
            chunk = samples[i:j]
            batched.append(chunk[0] if len(chunk) == 1 else self._stack_batch(chunk))
            i = j
        return iter(batched)

    def __len__(self):
        if self.batch_size == 1:
            return len(self.samples)
        # Conservative upper bound: ceil(n / batch_size)
        return math.ceil(len(self.samples) / self.batch_size)


class _DummyDataset:
    """Minimal dataset stub that only exposes __len__ (needed by dinv.Trainer)."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset(BaseDataset):
    """3D Walnut CT training dataset with parallel-read distributed loading.

    Trains jointly on all sparsity configs in `num_projs`. Returns a dict
    of physics factories and num_operators keyed by num_proj so the solver
    can select the correct physics for each (walnut_id, num_proj) sample.
    """

    name = "tomography_3d_train"

    requirements = [
        "torch",
        "numpy",
        "pandas",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    parameters = {
        # Absolute path to the Walnut-CBCT directory on the cluster
        "input_dir": ["/lustre/fswork/projects/rech/fio/commun/Walnut-CBCT"],
        # Sparsity configs to train on simultaneously
        "num_projs": [[30, 50, 100]],
        # Number of samples per gradient step (batch dim 0 of x / x_sparse / y_tl).
        "batch_size": [1],
        # Cap on training samples (None = use all available)
        "max_train_samples": [None],
        # Cap on validation samples (None = use all available)
        "max_val_samples": [None],
    }

    def __init__(
        self,
        input_dir="/lustre/fswork/projects/rech/fio/commun/Walnut-CBCT",
        num_projs=(30, 50, 100),
        batch_size=1,
        max_train_samples=None,
        max_val_samples=None,
    ):
        self.input_dir = Path(input_dir)
        self.num_projs = list(num_projs)
        self.batch_size = int(batch_size)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_csv(self, num_proj: int) -> pd.DataFrame:
        """Load dataset_{num_proj}p.csv and return a DataFrame."""
        csv_path = self.input_dir / f"dataset_{num_proj}p.csv"
        return pd.read_csv(csv_path)

    def _load_trajectory(self, sample_id: int, num_proj: int) -> torch.Tensor:
        """Load and subsample cone-beam trajectory for a given Walnut sample.

        The .geom file contains 1201 rows (last dropped); we uniformly subsample
        to num_proj angles via linspace over [0, 1200].
        """
        geom_path = self.input_dir / f"trajectory/Walnut{sample_id}.geom"
        traj = np.loadtxt(str(geom_path))
        traj = traj[:-1]  # drop last row → 1200 angles
        sparse_indexes = torch.linspace(0, 1201, steps=num_proj + 1, dtype=torch.long)[
            :-1
        ]  # (num_proj,)
        return torch.from_numpy(traj[sparse_indexes.numpy()].copy()).float()

    def _load_opnorm(
        self, sample_id: int, num_proj: int, opnorm_df: pd.DataFrame
    ) -> float:
        """Look up the pre-computed operator norm for sample_id and num_proj."""
        return float(opnorm_df.loc[opnorm_df["id"] == sample_id, str(num_proj)].item())

    # ------------------------------------------------------------------
    # Physics factory
    # ------------------------------------------------------------------

    def _make_physics_factory(
        self, trajectory_sparse: torch.Tensor, num_operators: int
    ):
        """Return an angle-split physics factory (one TomographyWithAstra per rank).

        The factory signature is factory(index, device, shared=None) as
        expected by deepinv.distributed.distribute().
        """
        num_proj = trajectory_sparse.shape[0]
        _base, _rem = divmod(num_proj, num_operators)
        _sizes = [_base + (1 if i < _rem else 0) for i in range(num_operators)]
        _cumsum = [0] + list(torch.cumsum(torch.tensor(_sizes), dim=0).tolist())
        traj_cpu = trajectory_sparse.cpu()

        def factory(index: int, device: torch.device, shared=None):
            import deepinv as dinv

            start = int(_cumsum[index])
            end = int(_cumsum[index + 1])
            traj_subset = traj_cpu[start:end].clone().to(device)
            print(
                f"[tomography_3d] physics_factory op {index}: angles [{start}:{end}]",
                flush=True,
            )
            return dinv.physics.TomographyWithAstra(
                img_size=(501, 501, 501),
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
    # Parallel sample loading — every rank reads independently
    # ------------------------------------------------------------------

    def _load_sample_by_id(
        self,
        maps: WalnutMemoryMaps,
        sample_id: int,
        num_proj: int,
        ctx,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a specific (walnut_id, num_proj) entry from this rank's memory maps.

        All tensors are kept on **CPU** so that the full dataset fits in system RAM
        regardless of the number of samples.  ``_unpack_batch`` in the solver moves
        each batch to the GPU just before the forward pass.

        Returns:
            x        : (1, 1, 501, 501, 501) ground-truth volume   [CPU]
            x_sparse : (1, 1, 501, 501, 501) FDK sparse recon      [CPU]
            y_full   : (1, 1, 972, num_proj, 768) sinogram          [CPU]
        """
        print(
            f"[rank {ctx.rank}] loading walnut id={sample_id}, num_proj={num_proj}...",
            flush=True,
        )

        x = (
            torch.as_tensor(maps.reference_rcs[num_proj][sample_id][:])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # → (1, 1, 501, 501, 501) on CPU
        )

        x_sparse = (
            torch.as_tensor(maps.sparse_rcs[num_proj][sample_id][:])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # → (1, 1, 501, 501, 501) on CPU
        )

        # sinogram already permuted to (972, num_proj, 768) by WalnutMemoryMaps
        y_full = (
            maps.sinogram[num_proj][sample_id]
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # → (1, 1, 972, num_proj, 768) on CPU
        )

        print(
            f"[rank {ctx.rank}] loaded: x={tuple(x.shape)}, "
            f"x_sparse={tuple(x_sparse.shape)}, y={tuple(y_full.shape)}",
            flush=True,
        )

        return x, x_sparse, y_full

    def _split_sinogram(self, y_full: torch.Tensor, num_operators: int) -> TensorList:
        """Split (1,1,972,num_proj,768) along the angle dim into a TensorList.

        Slicing the angle dimension produces non-contiguous views, so we call
        .contiguous() on CPU and then move
        each slice to the target device.
        """
        device = y_full.device
        y_cpu = y_full.cpu()
        num_angles = y_cpu.shape[3]
        _base, _rem = divmod(num_angles, num_operators)
        _sizes = [_base + (1 if i < _rem else 0) for i in range(num_operators)]
        _cumsum = [0] + list(torch.cumsum(torch.tensor(_sizes), dim=0).tolist())
        return TensorList(
            [
                y_cpu[:, :, :, int(_cumsum[i]) : int(_cumsum[i + 1]), :]
                .contiguous()
                .to(device)
                for i in range(num_operators)
            ]
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_data(self):
        """Load train/val dataloaders + per-config physics factories.

        Returns a dict with keys expected by Objective.set_data():
          train_dataloader : ParallelReadDataLoader (shuffled, all configs)
          val_dataloader   : ParallelReadDataLoader (ordered, all configs)
          physics          : dict {num_proj: factory_fn}
          num_operators    : dict {num_proj: int}
          ground_truth_shape, min_pixel, max_pixel, operator_norm
        """
        # Export distributed env from submitit if running inside a Slurm job
        # and RANK/WORLD_SIZE are not yet set.
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            try:
                import submitit  # noqa: PLC0415

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
                print("[tomography_3d] distributed env exported via submitit")
            except (ImportError, RuntimeError) as e:
                print(f"[tomography_3d] non-distributed mode ({e})")
        else:
            print(
                f"[tomography_3d] RANK={os.environ['RANK']}, "
                f"WORLD_SIZE={os.environ['WORLD_SIZE']}"
            )

        # cleanup=False: keep the process group alive so the solver's
        # DistributedContext (created later in run()) reuses it.
        with DistributedContext(seed=42, cleanup=False) as ctx:
            rank = ctx.rank
            print(
                f"[tomography_3d] rank {rank}/{ctx.world_size}, "
                f"num_projs={self.num_projs}",
                flush=True,
            )

            # 1. Load opnorms CSV once (shared across configs).
            try:
                opnorm_df = pd.read_csv(self.input_dir / "opnorms.csv")
            except FileNotFoundError:
                opnorm_df = None
                print("[tomography_3d] opnorms.csv not found — using 1.0 for all")

            # 2. Build per-config physics factories (from first training sample's
            #    trajectory for each num_proj).  num_operators == num_proj so each
            #    angle gets its own sub-operator.
            physics_dict: dict = {}
            num_operators_dict: dict = {}
            for num_proj in self.num_projs:
                df = self._load_csv(num_proj)
                first_id = int(df[df["split_set"] == "train"].iloc[0]["id"])
                traj = self._load_trajectory(first_id, num_proj).to(ctx.device)
                num_ops = num_proj  # one sub-operator per projection angle
                physics_dict[num_proj] = self._make_physics_factory(traj, num_ops)
                num_operators_dict[num_proj] = num_ops
                print(
                    f"[tomography_3d] built physics factory for num_proj={num_proj}, "
                    f"num_operators={num_ops}",
                    flush=True,
                )

            # 3. Build memory-map handles on every rank.
            #    When max_*_samples is set, limit the number of walnut IDs opened
            #    per config to ceil(cap / len(num_projs)) so loading is fast.
            def _max_per_config(total_cap):
                if total_cap is None:
                    return None
                # Each walnut ID generates len(num_projs) samples (one per config),
                # so we only need ceil(cap / len(num_projs)) IDs per config.
                return math.ceil(total_cap / len(self.num_projs))

            print(f"[rank {rank}] opening memory maps...", flush=True)
            train_maps = WalnutMemoryMaps(
                self.input_dir,
                self.num_projs,
                split="train",
                max_per_config=_max_per_config(self.max_train_samples),
            )
            val_maps = WalnutMemoryMaps(
                self.input_dir,
                self.num_projs,
                split="validation",
                max_per_config=_max_per_config(self.max_val_samples),
            )
            print(f"[rank {rank}] memory maps ready", flush=True)

            # 4. Load all (walnut_id, num_proj) combinations.
            #    Train: 29 walnuts × len(num_projs) configs.
            #    Val  :  4 walnuts × len(num_projs) configs.
            def _load_split(maps: WalnutMemoryMaps, split: str):
                """Return flat list of (x, x_sparse, y_tl, num_proj) tuples and opnorms.

                For every walnut ID, all num_proj configs are loaded in order:
                    (id1, 30p), (id1, 50p), (id1, 100p), (id2, 30p), ...
                This keeps all configs for one walnut ID in a consecutive group of
                size len(num_projs), which pairs naturally with grad_accumulation_steps
                and group-level shuffling.
                """
                samples = []
                norms = []
                # All configs have the same IDs loaded (same max_per_config).
                all_ids = [int(k) for k in maps.reference_rcs[self.num_projs[0]].keys()]
                print(
                    f"[tomography_3d] loading {len(all_ids)} {split} IDs × "
                    f"{len(self.num_projs)} configs = "
                    f"{len(all_ids) * len(self.num_projs)} samples "
                    f"from {self.num_projs}",
                    flush=True,
                )
                for sid in all_ids:
                    for assigned_np in self.num_projs:
                        num_ops = num_operators_dict[assigned_np]
                        x, x_sparse, y_full = self._load_sample_by_id(
                            maps, sid, assigned_np, ctx
                        )
                        samples.append(
                            (
                                x,
                                x_sparse,
                                self._split_sinogram(y_full, num_ops),
                                assigned_np,
                            )
                        )
                        if opnorm_df is not None:
                            try:
                                norms.append(
                                    self._load_opnorm(sid, assigned_np, opnorm_df)
                                )
                            except Exception as e:
                                print(
                                    f"[tomography_3d] opnorm missing for "
                                    f"id={sid}, num_proj={assigned_np}: {e}"
                                )
                                norms.append(1.0)
                        else:
                            norms.append(1.0)
                return samples, norms

            train_samples, train_norms = _load_split(train_maps, "train")
            val_samples, val_norms = _load_split(val_maps, "validation")

            # Optional caps (useful for quick smoke-tests without full data load)
            if self.max_train_samples is not None:
                train_samples = train_samples[: self.max_train_samples]
                train_norms = train_norms[: self.max_train_samples]
                print(
                    f"[tomography_3d] train capped at {len(train_samples)} samples",
                    flush=True,
                )
            if self.max_val_samples is not None:
                val_samples = val_samples[: self.max_val_samples]
                print(
                    f"[tomography_3d] val capped at {len(val_samples)} samples",
                    flush=True,
                )

            print(
                f"[tomography_3d] loaded {len(train_samples)} train / "
                f"{len(val_samples)} val samples total",
                flush=True,
            )

            # 5. Barrier — wait for all ranks to finish reading.
            if ctx.use_dist:
                print(
                    f"[rank {rank}] data loading done, waiting at barrier...",
                    flush=True,
                )
                torch.distributed.barrier()
                print(f"[rank {rank}] barrier passed", flush=True)

            # Use the mean operator norm over all training (id, num_proj) pairs.
            operator_norm = float(np.mean(train_norms)) if train_norms else 1.0
            print(
                f"[tomography_3d] operator_norm (mean over train)={operator_norm:.4f}",
                flush=True,
            )

            x0 = train_samples[0][0]
            ground_truth_shape = tuple(x0.shape)  # (1, 1, 501, 501, 501)
            x0_cpu = x0.cpu()  # avoid CUDA kernel dispatch for stats
            min_pixel = float(x0_cpu.min().item())
            max_pixel = float(x0_cpu.max().item())

            print(
                f"[tomography_3d] done. shape={ground_truth_shape}, "
                f"pixel=[{min_pixel:.3f},{max_pixel:.3f}], "
                f"num_projs={self.num_projs}",
                flush=True,
            )

        return dict(
            # Train loader is shuffled: each epoch draws IID (walnut_id, num_proj) pairs.
            train_dataloader=ParallelReadDataLoader(
                train_samples,
                shuffle=True,
                batch_size=self.batch_size,
                group_size=len(self.num_projs),
            ),
            val_dataloader=ParallelReadDataLoader(
                val_samples,
                shuffle=False,
                batch_size=self.batch_size,
            ),
            # physics and num_operators are dicts keyed by num_proj.
            physics=physics_dict,
            num_operators=num_operators_dict,
            ground_truth_shape=ground_truth_shape,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
            operator_norm=operator_norm,
        )
