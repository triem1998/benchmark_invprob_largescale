"""Tomography utilities: Walnut dataset classes, sinogram utilities, and helpers."""

from __future__ import annotations

import math as _math
import random as _pyrandom
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Optional as _Opt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import Sampler as _Sampler
from tqdm import tqdm


class WalnutMemoryMaps:
    """Open memory-mapped files for all (num_proj, sample_id) combinations.

    This is a plain storage object — no torch.Dataset interface, no batching,
    no random sampling.  It simply exposes three nested dicts:

        reference_rcs[num_proj][sample_id]  — ground-truth volume  (501,501,501)
        sparse_rcs[num_proj][sample_id]     — FDK sparse recon      (501,501,501)
        sinogram[num_proj][sample_id]       — ASTRA sinogram (972,num_proj,768)

    Sinograms are subsampled from 1200 angles to num_proj via uniform linspace
    and permuted to ASTRA (H, P, W) format on construction.
    """

    def __init__(
        self,
        input_dir: Path,
        num_projs: list[int],
        split: str,  # "train", "validation", or "test"
        max_per_config: int | None = None,  # limit walnut IDs loaded per config
    ):
        self.reference_rcs: dict = {}
        self.sparse_rcs: dict = {}
        self.sinogram: dict = {}

        print(
            f"[WalnutMemoryMaps] opening memory maps for split={split!r}...", flush=True
        )
        import tensordict

        for num_proj in num_projs:
            csv_path = input_dir / f"dataset_{num_proj}p.csv"
            df = pd.read_csv(csv_path)
            df = df.loc[df["split_set"] == split].reset_index(drop=True)
            if max_per_config is not None:
                df = df.iloc[:max_per_config].reset_index(drop=True)

            sparse_indexes = torch.linspace(
                0, 1201, steps=num_proj + 1, dtype=torch.long
            )[
                :-1
            ]  # (num_proj,) — uniform subsampling of 1200 angles

            self.reference_rcs[num_proj] = {}
            self.sparse_rcs[num_proj] = {}
            self.sinogram[num_proj] = {}

            for row in tqdm(
                df.itertuples(), total=len(df), desc=f"{num_proj}p {split}"
            ):
                sid = row.id
                shape = (row.number_of_slice, row.num_voxels, row.num_voxels)

                self.reference_rcs[num_proj][sid] = (
                    tensordict.MemoryMappedTensor.from_filename(
                        input_dir / row.reconstruction_file,
                        shape=shape,
                        dtype=torch.float32,
                    )
                )
                self.sparse_rcs[num_proj][sid] = (
                    tensordict.MemoryMappedTensor.from_filename(
                        input_dir / row.sparse_reconstruction_file,
                        shape=shape,
                        dtype=torch.float32,
                    )
                )
                # Sinogram: (1200, 972, 768) → subsample → permute to (972, num_proj, 768)
                raw = torch.from_numpy(
                    np.memmap(
                        input_dir / row.sinogram_file,
                        dtype="float32",
                        mode="c",
                        shape=(1200, 972, 768),
                    )
                )
                self.sinogram[num_proj][sid] = (
                    raw[sparse_indexes].permute(1, 0, 2).contiguous()
                )

        print("[WalnutMemoryMaps] memory maps ready.", flush=True)


# ── Standard DataLoader-compatible classes ─────────────────────────────────
# (demo_tomo-style: CSV + np.memmap, lazy per-sample loading)


@_dataclass(frozen=True)
class WalnutEntry:
    """Metadata for one (walnut_id, num_proj) sample; no binary data is loaded yet."""

    sample_id: int
    num_proj: int
    number_of_slice: int
    num_voxels: int
    reconstruction_file: str
    sparse_reconstruction_file: str
    sinogram_file: str


class WalnutGroupSampler(_Sampler):
    """Shuffle at the walnut-ID group level so that all num_proj configs for the same
    walnut appear consecutively — aligning naturally with ``grad_accumulation_steps``.

    Call ``set_epoch(epoch)`` before each epoch to get reproducible but distinct
    orderings across epochs (seed + epoch).
    """

    def __init__(self, dataset: "WalnutTomoDataset", shuffle: bool, seed: int = 0):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        n = len(self.dataset)
        g = self.dataset.group_size
        if n == 0:
            return iter([])
        if not self.shuffle:
            return iter(range(n))
        rng = _pyrandom.Random(self.seed + self.epoch)
        n_groups = _math.ceil(n / g)
        group_ids = list(range(n_groups))
        rng.shuffle(group_ids)
        order = []
        for gid in group_ids:
            start = gid * g
            end = min(start + g, n)
            group = list(range(start, end))
            rng.shuffle(group)
            order.extend(group)
        return iter(order)


class WalnutTomoDataset(_Dataset):
    """PyTorch Dataset for the Walnut-CBCT dataset using CSV metadata + np.memmap.

    Each ``__getitem__`` reads one (walnut_id, num_proj) sample lazily from disk
    (no upfront data copy into RAM).  Items are returned as a dict with keys:
    ``"x"``, ``"x_sparse"``, ``"y_full"``, ``"num_proj"``, ``"sample_id"``,
    ``"operator_norm"``.

    Parameters
    ----------
    input_dir : str or Path
        Root directory of the Walnut-CBCT dataset.
    num_projs : iterable of int
        Sparsity configs to include (e.g. ``[30, 50, 100]``).
    split : str
        One of ``"train"``, ``"validation"``, ``"test"``.
    max_samples : int or None
        Optional cap on the total number of (walnut_id, num_proj) entries.
    """

    def __init__(self, input_dir, num_projs, split: str, max_samples=None):
        self.input_dir = Path(input_dir)
        self.num_projs = tuple(int(v) for v in num_projs)
        self.split = str(split)
        # group_size = number of num_proj configs per walnut ID
        self.group_size = len(self.num_projs)
        self.entries: list[WalnutEntry] = self._build_entries(max_samples)
        self.operator_norm_map: dict = self._load_operator_norm_map()
        # Precompute angle-subsampling indices for each num_proj
        self._sparse_indices = {
            np_: torch.linspace(0, 1201, steps=np_ + 1, dtype=torch.long)[:-1].numpy()
            for np_ in self.num_projs
        }

    # ------------------------------------------------------------------
    # Entry building
    # ------------------------------------------------------------------

    def _build_entries(self, max_samples: _Opt[int]) -> list:
        frames: dict = {}
        for num_proj in self.num_projs:
            csv_path = self.input_dir / f"dataset_{num_proj}p.csv"
            df = pd.read_csv(csv_path)
            df = df.loc[df["split_set"] == self.split].reset_index(drop=True)
            if max_samples is not None:
                per_cfg = _math.ceil(max_samples / max(1, len(self.num_projs)))
                df = df.iloc[:per_cfg].reset_index(drop=True)
            frames[num_proj] = df

        common_ids = sorted(
            set.intersection(
                *[set(df["id"].astype(int).tolist()) for df in frames.values()]
            )
        )
        indexed = {np_: df.set_index("id", drop=False) for np_, df in frames.items()}
        entries = []
        for sample_id in common_ids:
            for num_proj in self.num_projs:
                row = indexed[num_proj].loc[int(sample_id)]
                entries.append(
                    WalnutEntry(
                        sample_id=int(sample_id),
                        num_proj=int(num_proj),
                        number_of_slice=int(row["number_of_slice"]),
                        num_voxels=int(row["num_voxels"]),
                        reconstruction_file=str(row["reconstruction_file"]),
                        sparse_reconstruction_file=str(
                            row["sparse_reconstruction_file"]
                        ),
                        sinogram_file=str(row["sinogram_file"]),
                    )
                )
        if max_samples is not None:
            entries = entries[: int(max_samples)]
        return entries

    def _load_operator_norm_map(self) -> dict:
        """Load per-(sample_id, num_proj) operator norms from opnorms.csv."""
        opnorm_path = self.input_dir / "opnorms.csv"
        if not opnorm_path.exists():
            return {}
        df = pd.read_csv(opnorm_path).set_index("id")
        result: dict = {}
        for e in self.entries:
            col = str(e.num_proj)
            try:
                result[(e.sample_id, e.num_proj)] = float(df.loc[e.sample_id, col])
            except Exception:
                result[(e.sample_id, e.num_proj)] = 1.0
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Load one sample and return a dict of (C, ...) tensors.

        Tensors have shape ``(1, D, H, W)`` for volumes and ``(1, H, P, W)``
        for sinograms (channel-first, no batch dim — DataLoader collation
        adds the batch dimension, yielding ``(B, C, ...)`` for the model).

        Returns
        -------
        dict with keys: ``x``, ``x_sparse``, ``y_full``, ``num_proj``,
        ``sample_id``, ``operator_norm``.
        """
        entry = self.entries[idx]
        vol_shape = (
            entry.number_of_slice,
            entry.num_voxels,
            entry.num_voxels,
        )
        x_mm = np.memmap(
            self.input_dir / entry.reconstruction_file,
            dtype="float32",
            mode="r",
            shape=vol_shape,
        )
        x_sparse_mm = np.memmap(
            self.input_dir / entry.sparse_reconstruction_file,
            dtype="float32",
            mode="r",
            shape=vol_shape,
        )
        y_mm = np.memmap(
            self.input_dir / entry.sinogram_file,
            dtype="float32",
            mode="r",
            shape=(1200, 972, 768),
        )
        proj_idx = self._sparse_indices[entry.num_proj]
        y_sparse = np.asarray(y_mm[proj_idx]).transpose(1, 0, 2).copy()
        x = torch.from_numpy(np.asarray(x_mm).copy()).unsqueeze(0)  # (1, D, H, W)
        x_sp = torch.from_numpy(np.asarray(x_sparse_mm).copy()).unsqueeze(0)
        y_full = torch.from_numpy(y_sparse).unsqueeze(0)  # (1, H, P, W)
        op_norm = self.operator_norm_map.get((entry.sample_id, entry.num_proj), 1.0)
        # Return (C, ...) tensors without a batch dim — DataLoader adds it via
        # collation, so the final batch shape is (B, C, D, H, W) as expected.
        return {
            "x": x.float(),  # (1, D, H, W)
            "x_sparse": x_sp.float(),
            "y_full": y_full.float(),  # (1, H, P, W)
            "num_proj": int(entry.num_proj),
            "sample_id": int(entry.sample_id),
            "operator_norm": float(op_norm),
        }


def load_trajectory_sparse(input_dir, sample_id: int, num_proj: int) -> torch.Tensor:
    """Load and uniformly subsample a cone-beam trajectory for a Walnut sample.

    The .geom file contains 1201 rows (last dropped); ``num_proj`` angles are
    selected via ``torch.linspace(0, 1201, num_proj+1)``.

    Returns
    -------
    Tensor of shape (num_proj, 12)  — float32, on CPU.
    """
    geom_path = Path(input_dir) / f"trajectory/Walnut{int(sample_id)}.geom"
    traj = np.loadtxt(str(geom_path))
    traj = traj[:-1]  # drop last row → 1200 angles
    sparse_indexes = torch.linspace(0, 1201, steps=num_proj + 1, dtype=torch.long)[:-1]
    return torch.from_numpy(traj[sparse_indexes.numpy()].copy()).float()


# ── Sinogram splitting & evaluation helpers ───────────────────────────────────


def projection_splits(num_angles: int, num_operators: int) -> list:
    """Divide ``num_angles`` projection angles evenly among ``num_operators`` sub-operators.

    Returns a list of ``(start, end)`` half-open index pairs (one per sub-operator).
    """
    base, rem = divmod(int(num_angles), int(num_operators))
    sizes = [base + (1 if i < rem else 0) for i in range(num_operators)]
    edges = [0]
    for s in sizes:
        edges.append(edges[-1] + s)
    return [(edges[i], edges[i + 1]) for i in range(num_operators)]


def split_sinogram(y_full, num_operators: int):
    """Split a sinogram along the angle axis into a TensorList.

    Parameters
    ----------
    y_full : Tensor  (B, 1, H, num_proj, W)
        Full sinogram where dim 3 is the angle axis.
    num_operators : int
        Number of splits to create (one per distributed sub-operator).

    Returns
    -------
    TensorList
        ``num_operators`` contiguous angle-split tensors.
    """
    from deepinv.utils.tensorlist import TensorList

    chunks = projection_splits(int(y_full.shape[3]), int(num_operators))
    return TensorList([y_full[:, :, :, s:e, :].contiguous() for s, e in chunks])


def ensure_dir(path) -> Path:
    """Create ``path`` (and any parents) if it does not exist; return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def append_metrics_row(path, row: dict) -> None:
    """Append one row to a CSV file, writing the header on first write.

    Parameters
    ----------
    path : str or Path
        Destination CSV (parent directories are created as needed).
    row : dict
        Mapping from column name to value.
    """
    import csv

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ── DataLoader collate helpers ──────────────────────────────────────────────


def collate_deepinv_batch(batch):
    """Collate function for deepinv batches that may contain TensorList measurements.

    Handles both single-operator (plain tensor) and multi-operator (TensorList)
    measurements returned by deepinv stacked physics datasets.

    Parameters
    ----------
    batch : list of tuples
        Each element is ``(ground_truth, measurement)``.

    Returns
    -------
    tuple
        ``(ground_truth_batch, measurement_batch)``
    """
    from deepinv.utils.tensorlist import TensorList

    if len(batch) == 1:
        ground_truth, measurement = batch[0]
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.unsqueeze(0)
        if isinstance(measurement, TensorList):
            measurement = TensorList(
                [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
            )
        elif isinstance(measurement, (list, tuple)):
            measurement = [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
        elif measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        return ground_truth, measurement

    ground_truths, measurements = [], []
    for gt, meas in batch:
        ground_truths.append(gt)
        measurements.append(meas)

    ground_truth_batch = torch.stack(ground_truths, dim=0)

    if isinstance(measurements[0], TensorList):
        num_ops = len(measurements[0])
        measurement_batch = TensorList(
            [torch.stack([m[i] for m in measurements], dim=0) for i in range(num_ops)]
        )
    elif isinstance(measurements[0], (list, tuple)):
        num_ops = len(measurements[0])
        measurement_batch = [
            torch.stack([m[i] for m in measurements], dim=0) for i in range(num_ops)
        ]
    else:
        measurement_batch = torch.stack(measurements, dim=0)

    return ground_truth_batch, measurement_batch
