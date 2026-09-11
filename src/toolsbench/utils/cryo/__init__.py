"""Cryo-ET helpers for the equivariant-imaging training case.

Physics (both the astra and the pure-torch operator, plus angle sharding), the
icecream UNet3D, the two self-supervised loss terms, the rotation used by the
equivariance term, and the GPU FSC metric. The benchmark-facing pieces live in
``toolsbench.invprob.cryo_ei`` and ``toolsbench.solver.equivariant``.
"""

from .fsc import GpuFSC, fsc_resolution, fsc_shell
from .losses import EqLoss, ObsLoss, as_sinogram
from .models import build_unet3d
from .physics import (
    CryoEISpec,
    CryoPair,
    build_cryo_pair,
    resolve_num_operators,
    resolve_tomography_backend,
    split_sinogram,
)
from .transform import Rotate3D
from .utils import (
    AMP_DTYPES,
    AmpDenoiser,
    normalize_num_operators,
    reduce_metrics_max,
    wants_sharding,
)

__all__ = [
    "AMP_DTYPES",
    "AmpDenoiser",
    "CryoEISpec",
    "CryoPair",
    "EqLoss",
    "GpuFSC",
    "ObsLoss",
    "Rotate3D",
    "as_sinogram",
    "build_cryo_pair",
    "build_unet3d",
    "fsc_resolution",
    "fsc_shell",
    "resolve_num_operators",
    "resolve_tomography_backend",
    "normalize_num_operators",
    "reduce_metrics_max",
    "split_sinogram",
    "wants_sharding",
]
