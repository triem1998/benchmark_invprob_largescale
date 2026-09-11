"""Fourier Shell Correlation, on GPU.

GPU Fourier shell correlation (``fsc_shell``,
``fsc_resolution``, ``GpuFSC``).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["GpuFSC", "fsc_resolution", "fsc_shell"]


def fsc_shell(fsc_curve: np.ndarray, threshold: float) -> int:
    """Return first shell index where FSC drops below *threshold* (or last shell)."""
    below = np.where(fsc_curve < threshold)[0]
    return int(below[0]) if len(below) > 0 else int(len(fsc_curve) - 1)


def fsc_resolution(
    fsc_curve: np.ndarray, shape, pixel_size: float, threshold: float
) -> tuple[int, float, int]:
    """Return (shell, resolution_angstrom, n_ref) for a volume of *shape*.

    Shell ``k`` sits at spatial frequency ``k / n_ref`` cycles/voxel, so its
    resolution is ``n_ref * pixel_size / k`` angstrom.
    """
    n_ref = int(max(shape))
    k = fsc_shell(fsc_curve, threshold)
    return k, n_ref * pixel_size / max(k, 1), n_ref


class GpuFSC:
    """Fourier Shell Correlation computed entirely on GPU (float32).

    Works for any volume shape, cubic or not.  Shells are binned on the *true*
    spatial frequency radius: each axis index is divided by that axis' own
    length, so one step along a short axis (a large frequency step) is not
    confused with one step along a long axis (a small one)::

        f_i   = (idx_i - N_i // 2) / N_i        # cycles/voxel, in [-0.5, 0.5)
        rho   = sqrt(fz^2 + fy^2 + fx^2)
        shell = round(rho * n_ref),  n_ref = max(D, H, W)

    For a cubic volume this reduces to the plain index radius, so cubic results
    are unchanged.  Shell maps are cached per shape, so one instance may be
    reused across volumes of differing shapes.

    Args:
        device: torch device string or object.
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self._cache: dict[tuple[int, ...], tuple[torch.Tensor, int]] = {}

    def _shell_map(self, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
        cached = self._cache.get(shape)
        if cached is not None:
            return cached

        axes = [
            (torch.arange(n, dtype=torch.float32, device=self.device) - n // 2) / n
            for n in shape
        ]
        grid = torch.meshgrid(*axes, indexing="ij")
        rho = torch.sqrt(sum(g * g for g in grid))

        shells = torch.round(rho * int(max(shape))).long().reshape(-1)
        rhomax = int(shells.max().item()) + 1

        self._cache[shape] = (shells, rhomax)
        return self._cache[shape]

    def __call__(self, vol1: torch.Tensor, vol2: torch.Tensor) -> np.ndarray:
        """Return FSC curve as 1-D numpy array (same format as ``FSC(a,b)[:,0]``)."""
        v1 = vol1.squeeze().to(self.device, dtype=torch.float32)
        v2 = vol2.squeeze().to(self.device, dtype=torch.float32)
        if v1.shape != v2.shape:
            raise ValueError(
                f"FSC needs matching shapes, got {tuple(v1.shape)} vs {tuple(v2.shape)}"
            )

        sh, rhomax = self._shell_map(tuple(v1.shape))

        F1 = torch.fft.fftshift(torch.fft.fftn(v1))
        F2 = torch.fft.fftshift(torch.fft.fftn(v2))

        cross = (F1 * F2.conj()).real.reshape(-1)
        pow1 = (F1.real**2 + F1.imag**2).reshape(-1)
        pow2 = (F2.real**2 + F2.imag**2).reshape(-1)

        # scatter_add_ silently ignores trailing src values when the index is
        # shorter, which would bin only part of the volume against wrong radii.
        if sh.numel() != cross.numel():
            raise RuntimeError(
                f"shell map has {sh.numel()} entries but volume has {cross.numel()}"
            )

        z_ = torch.zeros(rhomax, dtype=torch.float32, device=self.device)
        num = z_.clone().scatter_add_(0, sh, cross)
        den1 = z_.clone().scatter_add_(0, sh, pow1)
        den2 = z_.clone().scatter_add_(0, sh, pow2)

        denom = torch.sqrt(den1 * den2)
        fsc = torch.where(denom > 0.0, num / denom, torch.zeros_like(num))
        return fsc.cpu().numpy()
