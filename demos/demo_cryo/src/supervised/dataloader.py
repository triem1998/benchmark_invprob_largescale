from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CryoDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_train_vols: int | None = None  # None = use all remaining after val split
    max_val_vols: int = 1  # explicit number of validation volumes
    seed: int = 0
    target_shape: tuple[int, int, int] | None = (
        None  # (D, H, W) to resize to; None = no resize
    )


@dataclass
class CryoDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: "CryoDataset"
    val_dataset: "CryoDataset"


class CryoDataset(Dataset):
    """Paired supervised dataset: IsoNet-corrected input → Icecream target.

    Each item is a full 3-D volume pair ``(x, y)`` of shape ``(1, D, H, W)``.
    Volumes are read from disk on demand (no pre-loading) via ``mrcfile``.
    If ``target_shape`` is set, volumes are trilinearly resampled to that size.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        target_shape: tuple[int, int, int] | None = None,
    ) -> None:
        self.pairs = pairs  # [(input_path, target_path), ...]
        self.target_shape = target_shape  # (D, H, W) or None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp_path, tgt_path = self.pairs[idx]

        # DeepInv convention: x = ground truth, y = measurement (network input)
        # x = icecream target, y = corrected input
        y = torch.from_numpy(
            np.array(
                mrcfile.open(str(inp_path), permissive=True).data, dtype=np.float32
            )
        ).unsqueeze(
            0
        )  # (1, D, H, W)  corrected → network input
        x = torch.from_numpy(
            np.array(
                mrcfile.open(str(tgt_path), permissive=True).data, dtype=np.float32
            )
        ).unsqueeze(
            0
        )  # (1, D, H, W)  icecream → ground truth

        if self.target_shape is not None:
            # interpolate expects (B, C, D, H, W) → add batch dim, then remove
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            y = torch.nn.functional.interpolate(
                y.unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        # Per-volume global standardization (zero-mean, unit-std) — applied
        # independently to each modality, matching icecream's normalize_volume.
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        return x, y


def _discover_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Scan tomo_* subdirectories and return (corrected, icecream) path pairs."""
    pairs: list[tuple[Path, Path]] = []
    tomo_dirs = sorted(input_dir.glob("tomo_*"))

    if not tomo_dirs:
        raise FileNotFoundError(
            f"No tomo_* subdirectories found under {input_dir}. "
            "Check that input_dir points to the empiar-11058 folder."
        )

    for tomo_dir in tomo_dirs:
        corrected = list(tomo_dir.glob("*_corrected.mrc"))
        icecream = list(tomo_dir.glob("*icecream.mrc"))

        if not corrected:
            print(f"[data] WARNING: no *_corrected.mrc found in {tomo_dir}, skipping.")
            continue
        if not icecream:
            print(f"[data] WARNING: no *icecream.mrc found in {tomo_dir}, skipping.")
            continue

        # Take the first match if multiple (shouldn't happen in practice).
        pairs.append((corrected[0], icecream[0]))

    return pairs


def _shuffle_split(
    pairs: list[tuple[Path, Path]],
    n_val: int,
    seed: int,
    max_train: int | None,
) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_val = max(1, min(int(n_val), len(shuffled) - 1))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    if max_train is not None:
        train_pairs = train_pairs[: int(max_train)]

    return train_pairs, val_pairs


def _build_loader(
    dataset: CryoDataset,
    shuffle: bool,
    cfg: CryoDataConfig,
) -> DataLoader:
    kwargs: dict = {
        "dataset": dataset,
        "batch_size": int(cfg.batch_size),
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_cryo_dataloaders(cfg: CryoDataConfig) -> CryoDataBundle:
    input_dir = Path(cfg.input_dir)
    all_pairs = _discover_pairs(input_dir)

    if not all_pairs:
        raise RuntimeError("No valid (input, target) pairs discovered.")

    train_pairs, val_pairs = _shuffle_split(
        all_pairs,
        n_val=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        max_train=cfg.max_train_vols,
    )

    train_dataset = CryoDataset(train_pairs, target_shape=cfg.target_shape)
    val_dataset = CryoDataset(val_pairs, target_shape=cfg.target_shape)

    print(
        f"[data] discovered={len(all_pairs)} train={len(train_dataset)} val={len(val_dataset)}"
    )

    train_loader = _build_loader(train_dataset, shuffle=True, cfg=cfg)
    val_loader = _build_loader(val_dataset, shuffle=False, cfg=cfg)

    print("[data] loaders built")
    return CryoDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
