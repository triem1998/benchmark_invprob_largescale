"""Memory-mapped file container for the Walnut-CBCT dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tensordict
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
        split: str,          # "train", "validation", or "test"
        max_per_config: int | None = None,  # limit walnut IDs loaded per config
    ):
        self.reference_rcs: dict = {}
        self.sparse_rcs: dict = {}
        self.sinogram: dict = {}

        print(f"[WalnutMemoryMaps] opening memory maps for split={split!r}...", flush=True)
        for num_proj in num_projs:
            csv_path = input_dir / f"dataset_{num_proj}p.csv"
            df = pd.read_csv(csv_path)
            df = df.loc[df["split_set"] == split].reset_index(drop=True)
            if max_per_config is not None:
                df = df.iloc[:max_per_config].reset_index(drop=True)

            sparse_indexes = torch.linspace(
                0, 1201, steps=num_proj + 1, dtype=torch.long
            )[:-1]  # (num_proj,) — uniform subsampling of 1200 angles

            self.reference_rcs[num_proj] = {}
            self.sparse_rcs[num_proj] = {}
            self.sinogram[num_proj] = {}

            for row in tqdm(df.itertuples(), total=len(df), desc=f"{num_proj}p {split}"):
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
