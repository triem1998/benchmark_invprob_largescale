"""Training dataset for 3D Walnut cone-beam CT (distributed, parallel-read).

Data lives on the cluster at `input_dir` (e.g. /lustre/.../Walnut-CBCT).
Memory-mapped tensors are built via WalnutDatasetMultiP; individual samples are
loaded directly from the memory maps by sample_id (bypassing __getitem__, which
picks randomly and would not reliably return a specific walnut in training mode).

Loading strategy (parallel-read variant):
  - WalnutDatasetMultiP is initialised on *every* rank to build its own
    memory-map handles.
  - *Every* rank reads each sample directly from the maps simultaneously.

Tensor shapes per sample:
  x      : (1, 1, 501, 501, 501)   — ground-truth volume
  y_full : (1, 1, 972, num_proj, 768) — angle-subsampled sinogram (ASTRA format)
  After _split_sinogram: TensorList of num_operators tensors, each
      (1, 1, 972, angles_per_op, 768)

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from benchopt import BaseDataset
from deepinv.distributed import DistributedContext
from deepinv.utils.tensorlist import TensorList

# ---------------------------------------------------------------------------
# Minimal in-memory dataloader (no multiprocessing, safe for distributed use)
# ---------------------------------------------------------------------------


class ParallelReadDataLoader:
    """Simple iterable over pre-loaded (x, y_tensorlist) pairs.

    Replaces the multiprocessing DataLoader that caused hangs with
    WalnutDatasetMultiP.  Attributes expected by dinv.Trainer are present:
      - drop_last  (bool)
      - dataset    (object with __len__)
    Each item yielded is (x, y_tensorlist) where x has shape
    (1, 1, 501, 501, 501) — batch dim already present.
    """

    def __init__(self, samples: List):
        self.samples = samples
        self.drop_last = False
        self.dataset = _DummyDataset(len(samples))

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


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
    """3D Walnut CT training dataset with parallel-read distributed loading."""

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
        # Number of sparse projection angles (30, 50, or 100)
        "num_proj": [30],
        # How many train / val samples to actually load (subset of the CSV split).
        # Set to -1 to load all available samples in the split.
        "max_train_samples": [3],
        # WalnutDatasetMultiP hard-codes validation to id=12 only,
        # so max_val_samples is always effectively 1.
        "max_val_samples": [1],
        # Number of physics sub-operators (angle groups) for distributed training.
        "num_operators": [None],
    }

    def __init__(
        self,
        input_dir="/lustre/fswork/projects/rech/fio/commun/Walnut-CBCT",
        num_proj=30,
        max_train_samples=3,
        max_val_samples=1,
        num_operators=None,
    ):
        self.input_dir = Path(input_dir)
        self.num_proj = int(num_proj)
        self.max_train_samples = int(max_train_samples)
        self.max_val_samples = int(max_val_samples)
        self.num_operators = int(num_operators) if num_operators is not None else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_csv(self) -> pd.DataFrame:
        """Load dataset_{num_proj}p.csv and return a DataFrame."""
        csv_path = self.input_dir / f"dataset_{self.num_proj}p.csv"
        return pd.read_csv(csv_path)

    def _make_walnut_dataset(self, train: bool):
        """Instantiate WalnutDatasetMultiP to build memory-map handles."""
        from toolsbench.utils.walnut import WalnutDatasetMultiP  # noqa: PLC0415

        return WalnutDatasetMultiP(
            self.input_dir,
            [f"dataset_{self.num_proj}p.csv"],
            train=train,
            batch_size=1,
            num_projs=[self.num_proj],
            patch_size=(501, 501, 501),  # ignored when patch_training=False
            center_crop=False,
            dataset_size=400,  # only controls __len__, irrelevant here
            output=["target", "sinogram"],
            patch_training=False,
            input_clamp=False,
        )

    def _load_trajectory(self, sample_id: int) -> torch.Tensor:
        """Load and subsample cone-beam trajectory for a given Walnut sample."""
        geom_path = self.input_dir / f"trajectory/Walnut{sample_id}.geom"
        traj = np.loadtxt(str(geom_path))
        traj = traj[:-1]
        sparse_indexes = torch.linspace(
            0, 1201, steps=self.num_proj + 1, dtype=torch.long
        )[
            :-1
        ]  # (num_proj,)
        return torch.from_numpy(traj[sparse_indexes.numpy()].copy()).float()

    def _load_opnorm(self, sample_id: int, opnorm_df: pd.DataFrame) -> float:
        """Look up the pre-computed operator norm for sample_id and num_proj."""
        return float(
            opnorm_df.loc[opnorm_df["id"] == sample_id, str(self.num_proj)].item()
        )

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
                f"[tomography_3d_pr] physics_factory op {index}: angles [{start}:{end}]",
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
        self, walnut_dataset, sample_id: int, ctx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a specific walnut by sample_id directly from this rank's memory maps."""
        num_proj = self.num_proj
        device = ctx.device

        print(
            f"[rank {ctx.rank}] loading walnut id={sample_id} from memory maps...",
            flush=True,
        )

        # reference_rcs[num_proj][sample_id]: MemoryMappedTensor (501, 501, 501)
        x = (
            torch.as_tensor(walnut_dataset.reference_rcs[num_proj][sample_id][:])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # → (1, 1, 501, 501, 501)
            .to(device)
        )

        # sinogram[num_proj][sample_id]: already-permuted tensor (972, num_proj, 768)
        y_full = (
            walnut_dataset.sinogram[num_proj][sample_id]
            .float()
            .unsqueeze(0)
            .unsqueeze(0)  # → (1, 1, 972, num_proj, 768)
            .to(device)
        )

        print(
            f"[rank {ctx.rank}] loaded: x={tuple(x.shape)}, y={tuple(y_full.shape)}",
            flush=True,
        )

        return x, y_full

    def _split_sinogram(self, y_full: torch.Tensor, num_operators: int) -> TensorList:
        """Split (1,1,972,num_proj,768) along the angle dim into a TensorList."""
        num_angles = y_full.shape[3]
        _base, _rem = divmod(num_angles, num_operators)
        _sizes = [_base + (1 if i < _rem else 0) for i in range(num_operators)]
        _cumsum = [0] + list(torch.cumsum(torch.tensor(_sizes), dim=0).tolist())
        return TensorList(
            [
                y_full[:, :, :, int(_cumsum[i]) : int(_cumsum[i + 1]), :].contiguous()
                for i in range(num_operators)
            ]
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_data(self):
        """Load train/val dataloaders + physics factory for 3D Walnut CT.

        Returns a dict with keys expected by Objective.set_data():
          train_dataloader, val_dataloader, physics, ground_truth_shape,
          num_operators, min_pixel, max_pixel, operator_norm.
        """
        # Export distributed env from submitit if running inside a Slurm job
        # and RANK/WORLD_SIZE are not yet set (e.g. first call from benchopt runner).
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
            num_operators = (
                self.num_operators if self.num_operators is not None else ctx.world_size
            )
            print(
                f"[tomography_3d] rank {rank}/{ctx.world_size}, "
                f"num_operators={num_operators}",
                flush=True,
            )

            # 1. Load CSV and select train/val rows
            df = self._load_csv()
            train_rows = df[df["split_set"] == "train"].reset_index(drop=True)
            val_rows = df[df["split_set"] == "validation"].reset_index(drop=True)

            if self.max_train_samples > 0:
                train_rows = train_rows.head(self.max_train_samples)
            # WalnutDatasetMultiP hard-codes validation to id=12 (FIXME upstream)
            val_rows = val_rows.head(self.max_val_samples)

            print(
                f"[tomography_3d] {len(train_rows)} train / {len(val_rows)} val samples",
                flush=True,
            )

            # 2. Load opnorms CSV once
            try:
                opnorm_df = pd.read_csv(self.input_dir / "opnorms.csv")
            except FileNotFoundError:
                opnorm_df = None
                print("[tomography_3d] opnorms.csv not found — using 1.0 for all")

            # 3. Physics factory from the first training sample's trajectory.
            first_id = int(train_rows.iloc[0]["id"])
            trajectory_sparse = self._load_trajectory(first_id).to(ctx.device)
            physics_factory = self._make_physics_factory(
                trajectory_sparse, num_operators
            )

            # 4. Build memory-map handles via WalnutDatasetMultiP on every rank.
            #    Each rank has its own independent handles — no broadcast needed.
            print(
                f"[rank {rank}] initialising WalnutDatasetMultiP memory maps...",
                flush=True,
            )
            train_dataset = self._make_walnut_dataset(train=True)
            val_dataset = self._make_walnut_dataset(train=False)
            print(
                f"[rank {rank}] WalnutDatasetMultiP memory maps ready",
                flush=True,
            )

            # 5. Every rank reads each sample directly from its own memory maps.
            def _load_split(rows, walnut_dataset):
                samples = []
                norms = []
                for _, row in rows.iterrows():
                    sid = int(row["id"])
                    x, y_full = self._load_sample_by_id(walnut_dataset, sid, ctx)
                    samples.append((x, self._split_sinogram(y_full, num_operators)))
                    if opnorm_df is not None:
                        try:
                            norms.append(self._load_opnorm(sid, opnorm_df))
                        except Exception as e:
                            print(f"[tomography_3d] opnorm missing for id={sid}: {e}")
                            norms.append(1.0)
                    else:
                        norms.append(1.0)
                return samples, norms

            train_samples, train_norms = _load_split(train_rows, train_dataset)
            val_samples, _ = _load_split(val_rows, val_dataset)

            # 6. Barrier — wait for all ranks to finish reading before proceeding.
            if ctx.use_dist:
                print(
                    f"[rank {rank}] data loading done, waiting at barrier...",
                    flush=True,
                )
                torch.distributed.barrier()
                print(f"[rank {rank}] barrier passed", flush=True)

            # Use the mean operator norm over training samples
            operator_norm = float(np.mean(train_norms)) if train_norms else 1.0
            print(
                f"[tomography_3d] operator_norm (mean over train)={operator_norm:.4f}",
                flush=True,
            )

            x0 = train_samples[0][0]
            ground_truth_shape = tuple(x0.shape)  # (1, 1, 501, 501, 501)
            min_pixel = float(x0.min().item())
            max_pixel = float(x0.max().item())

            print(
                f"[tomography_3d_pr] done. shape={ground_truth_shape}, "
                f"pixel=[{min_pixel:.3f},{max_pixel:.3f}], "
                f"num_operators={num_operators}",
                flush=True,
            )

        return dict(
            train_dataloader=ParallelReadDataLoader(train_samples),
            val_dataloader=ParallelReadDataLoader(val_samples),
            physics=physics_factory,
            ground_truth_shape=ground_truth_shape,
            num_operators=num_operators,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
            operator_norm=operator_norm,
        )
