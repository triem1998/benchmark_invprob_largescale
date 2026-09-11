"""Synthetic cryo-ET half-set inverse problem for the ``tomo_ei`` case.

The synthetic stand-in for an EMPIAR-11830-like setup: one
volume, a single-axis tilt series, and the two interleaved half-sets (split1 /
split2) whose cross-consistency makes the method self-supervised. No MRC files
and no half-set discovery on disk — the volume is generated, the angles come
from the config, and both sinograms are simulated.
"""

from dataclasses import dataclass

import torch
from deepinv.physics import GaussianNoise

from toolsbench.data import DataConfig, SyntheticData
from toolsbench.invprob.base import (
    BaseInvProb,
    InvProb,
    InvProbConfig,
    build_problem_params,
)
from toolsbench.utils.cryo import CryoEISpec
from toolsbench.utils.cryo.physics import TOMO_ANGLE_SIGN, build_one_operator
from toolsbench.utils.cryo.physics.sharded import resolve_tomography_backend


@dataclass
class _CryoEIParams:
    #: Tilt count of EMPIAR-11830 (3 degrees over [-60, 60]).
    num_angles: int = 41
    tilt_min: float = -60.0
    tilt_max: float = 60.0
    noise_level: float = 0.1
    seed: int = 42
    #: Backend used to *simulate* the measurements. The solver resolves its own
    #: backend for training, so a run can simulate on astra and train on torch.
    tomography_backend: str = "auto"


class CryoEIInvProb(BaseInvProb):
    """Single-axis tilt series of a synthetic volume, split into two half-sets.

    The volume is generated in astra's ``(n_slices, n_rows, n_cols)`` order —
    ``(Y, Z, X)`` for this geometry, Y being the tilt axis — which is the order
    the operators expect, so nothing is permuted per call.
    """

    def get_invprob(self, invprob_config: InvProbConfig) -> InvProb:
        params = build_problem_params(_CryoEIParams, invprob_config.params)
        self._validate(params)

        device = torch.device(invprob_config.device)
        volume = self._ground_truth(invprob_config, device)
        volume_shape = tuple(int(s) for s in volume.shape[-3:])

        angles = torch.linspace(
            params.tilt_min,
            params.tilt_max,
            params.num_angles,
            dtype=torch.float32,
        )
        # Both halves carry the *whole* tilt series: split1/split2 are the even
        # and odd movie frames recorded at each tilt, not a split of the tilts
        # themselves, so they share one geometry and differ only in noise. The
        # EMPIAR-11830 files confirm it — angles_..._split1.tlt and
        # angles_..._split2.tlt hold identical 41-angle lists. Kept as two
        # fields because nothing in the format forces them to agree.
        spec = CryoEISpec(
            volume_shape=volume_shape,
            angles_evn=angles,
            angles_odd=angles,
            angle_sign=TOMO_ANGLE_SIGN,
        )

        backend = resolve_tomography_backend(params.tomography_backend, device)
        measurements = [
            self._simulate(spec, half_angles, volume, device, params, backend, index)
            for index, half_angles in enumerate((spec.angles_evn, spec.angles_odd))
        ]

        return InvProb(
            ground_truth=volume,
            measurements=measurements,
            physics=spec,
            ground_truth_shape=volume.shape,
            num_operators=1,
            min_pixel=volume.min().item(),
            max_pixel=volume.max().item(),
        )

    def _ground_truth(self, invprob_config: InvProbConfig, device) -> torch.Tensor:
        data = SyntheticData().get_data(
            DataConfig(
                size=tuple(int(s) for s in invprob_config.size),
                batch_size=invprob_config.batch_size,
                channels=1,
                data_type=invprob_config.data_type,
                device=device,
                data_path=invprob_config.data_path,
            )
        )
        volume = data["data"]
        # Z-normalised: a non-zero mean projects to a
        # constant offset in A(x) that a centred sinogram can never match.
        return (volume - volume.mean()) / (volume.std() + 1e-8)

    def _simulate(self, spec, half_angles, volume, device, params, backend, index):
        """One half's noisy, z-normalised sinogram, from an unsharded operator."""
        physics = build_one_operator(
            spec, half_angles.to(device), device, num_operators=None, backend=backend
        )
        rng = torch.Generator(device=device).manual_seed(params.seed + index)
        physics.noise_model = GaussianNoise(sigma=params.noise_level, rng=rng)
        with torch.no_grad():
            sinogram = physics(volume)
        return (sinogram - sinogram.mean()) / (sinogram.std() + 1e-8)

    def _validate(self, params: _CryoEIParams) -> None:
        if params.num_angles < 2:
            raise ValueError("num_angles must be at least 2.")
        if params.tilt_max <= params.tilt_min:
            raise ValueError("tilt_max must be greater than tilt_min.")
        if params.noise_level < 0:
            raise ValueError("noise_level must be non-negative.")
