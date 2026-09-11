"""The two self-supervised loss terms, for ``tomo_ei`` and ``unrolled``.

``A`` is volume -> sinogram, not a frequency mask. The halves are a *dose* split, so
both terms are Noise2Noise::

    L_obs = MSE(c_odd * A_odd(f(evn)), y_odd) + MSE(c_evn * A_evn(f(odd)), y_evn)
    L_eq  = MSE(f(P_evn(x_rot)), y_rot)       + MSE(f(P_odd(y_rot)), x_rot)

``L_eq`` shares one rotation and targets each half on the *other* half's
reconstruction, keeping the target's noise independent.

The solver splits ``y`` to match the shards, so *both* operands go through
:func:`as_sinogram` before any statistic is taken.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.utils.tensorlist import TensorList
from torch.utils.checkpoint import checkpoint

__all__ = ["EqLoss", "ObsLoss", "as_sinogram"]


def as_sinogram(projection) -> torch.Tensor:
    """Reassemble a sharded ``A(x)`` into one ``(B, C, V, A, N)`` sinogram.

    Shards are contiguous and ascending, so concatenating on the angle axis
    rebuilds exactly what the unsharded operator would produce — which is what
    makes the loss independent of ``num_operators``.
    """
    return (
        projection
        if torch.is_tensor(projection)
        else torch.cat(list(projection), dim=3)
    )


def _ramp_half(t: torch.Tensor) -> torch.Tensor:
    """``sqrt(|k|)`` along the detector axis; the caller squares. Unit mean
    square, so a white residual keeps its scale. ``w(0) = 0``."""
    n = t.shape[-1]
    w = torch.fft.rfftfreq(n, device=t.device, dtype=torch.float32).abs().sqrt()
    w = w / w.pow(2).mean().sqrt().clamp_min(1e-12)
    return torch.fft.irfft(torch.fft.rfft(t.float(), dim=-1) * w, n=n, dim=-1).to(
        t.dtype
    )


class ObsLoss:
    """Cross half-set data fidelity.

    :param float weight: loss weight.
    :param str gain: scale calibration, one of :attr:`GAINS`. Volume and
        sinogram are z-normalised independently, so ``A(x_net)`` and ``y`` are
        not on one scale. ``znorm`` normalises both in-graph; the least-squares
        variants fit ``<a,y>/<a,a>``, refit each step or frozen after the first.
    :param bool ramp: weight the residual by ``|k|``; without it blur is cheap.
    """

    #: Accepted ``obs_gain`` values.
    GAINS = ("none", "znorm", "leastsq_xnet", "leastsq_xnet_frozen")

    def __init__(
        self, weight: float = 1.0, gain: str = "none", ramp: bool = False
    ) -> None:
        if gain not in self.GAINS:
            raise ValueError(f"obs_gain must be one of {self.GAINS}, got {gain!r}.")
        self.weight = weight
        self.gain = gain
        self.ramp = ramp
        self._gain_cache = None  # (init_evn, init_odd, c_odd, c_evn)

    def _gains(self, pair, x, y, a_odd_net, a_evn_net):
        """Least-squares ``c`` per half, under ``no_grad``.

        ``leastsq_xnet_frozen`` keys its cache on the ``init_*`` tensors. One
        volume here, so the key never changes and "frozen" means "after the
        first step"; the keying is kept for a future multi-volume dataset.
        """
        if self.gain == "none":
            return 1.0, 1.0
        c = self._gain_cache
        if (
            c is not None
            and self.gain == "leastsq_xnet_frozen"
            and c[0] is pair.init_evn
            and c[1] is pair.init_odd
        ):
            return c[2], c[3]
        with torch.no_grad():
            c_odd = (a_odd_net * y).sum() / ((a_odd_net * a_odd_net).sum() + 1e-8)
            c_evn = (a_evn_net * x).sum() / ((a_evn_net * a_evn_net).sum() + 1e-8)
        if self.gain == "leastsq_xnet_frozen":
            self._gain_cache = (pair.init_evn, pair.init_odd, c_odd, c_evn)
        return c_odd, c_evn

    def __call__(self, pair, x_net, y_net, y_evn, y_odd) -> torch.Tensor:
        # Both operands reassembled — see the module docstring.
        x, y = as_sinogram(y_evn), as_sinogram(y_odd)
        # Projected first so the gains reuse these rather than re-running A.
        a_odd = as_sinogram(pair.physics_odd.A(x_net))
        a_evn = as_sinogram(pair.physics_evn.A(y_net))
        if self.gain == "znorm":
            # In-graph: c is not the least-squares optimum, so detaching
            # would change the gradient.
            zn = lambda t: (t - t.mean()) / (t.std() + 1e-8)  # noqa: E731
            r_odd, r_evn = zn(a_odd) - zn(y), zn(a_evn) - zn(x)
        else:
            c_odd, c_evn = self._gains(pair, x, y, a_odd, a_evn)
            r_odd, r_evn = c_odd * a_odd - y, c_evn * a_evn - x
        if self.ramp:
            r_odd, r_evn = _ramp_half(r_odd), _ramp_half(r_evn)
        return self.weight * ((r_odd**2).mean() + (r_evn**2).mean())


class EqLoss:
    """Equivariance under a rotation, re-simulated through the real geometry.

    :param Rotate3D transform: shape-preserving rotation sampler.
    :param float weight: loss weight.
    :param bool unrolled: ``f`` is the PGD net, not a plain denoiser.
    :param float noise: multiple of the measured half-set noise level added to
        the simulated measurement.
    :param bool scale_free: z-normalise both operands, so Eq scores shape and
        ObsLoss owns amplitude.
    """

    #: Reduced over batch/channel and both detector axes, keeping the tilt-angle
    #: axis of a ``(B, C, V, A, N)`` sinogram.
    _PER_ANGLE = (0, 1, 2, 4)

    def __init__(
        self,
        transform,
        weight: float = 1.0,
        unrolled: bool = False,
        noise: float = 0.0,
        scale_free: bool = False,
    ) -> None:
        self._transform = transform
        self.weight = weight
        self.unrolled = unrolled
        self.noise = noise
        self.scale_free = scale_free
        self._criteria = nn.MSELoss(reduction="mean")

    def _noise_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Noise-to-signal ratio per tilt angle, from the half-set difference.

        ``x - y`` cancels the object, leaving ``2 sigma^2``. Dimensionless, so
        it transfers to ``y_sim``'s scale. Per angle: high tilts are noisier.
        """
        var_n = (x - y).var(dim=self._PER_ANGLE, keepdim=True) / 2.0
        var_s = (y.var(dim=self._PER_ANGLE, keepdim=True) - var_n).clamp_min(1e-12)
        return (var_n / var_s).sqrt()

    def _add_noise(self, y_sim, ratio: torch.Tensor):
        """``y_sim + eps``, ``eps`` scaled to ``y_sim``'s own per-angle std.

        A fixed sigma would let a louder model face less relative noise. Each
        shard slices ``ratio`` by its own angle count.
        """

        def _one(t, r):
            return t + self.noise * r * t.std(
                dim=self._PER_ANGLE, keepdim=True
            ) * torch.randn_like(t)

        if torch.is_tensor(y_sim):
            return _one(y_sim, ratio)
        out, a0 = [], 0
        for part in y_sim:
            a1 = a0 + part.shape[3]
            out.append(_one(part, ratio[..., a0:a1, :]))
            a0 = a1
        return TensorList(out)

    def _mse(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE, scale-free when asked. Checkpointed: the z-norm otherwise pins
        eight full volumes, recomputed for ~0.15% of a step."""
        if not self.scale_free:
            return self._criteria(est, target)
        return checkpoint(self._mse_scale_free, est, target, use_reentrant=False)

    def _mse_scale_free(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """In-graph: a detached z-norm is value-invariant but not
        gradient-invariant, leaving the shrink channel open."""
        zn = lambda t: (t - t.mean()) / (t.std() + 1e-8)  # noqa: E731
        return self._criteria(zn(est), zn(target))

    def _recon(self, v_rot: torch.Tensor, physics, model, ratio) -> torch.Tensor:
        """``f`` applied to a measurement simulated from ``v_rot``.

        The unrolled net also takes the sinogram. Noise goes in before ``fbp``.
        """
        y_sim = physics.A(v_rot)
        if ratio is not None:
            y_sim = self._add_noise(y_sim, ratio)
        init = physics.fbp(y_sim)
        if self.unrolled:
            return model(y_sim, physics, init=init)
        return model(init)

    def __call__(self, pair, model, x_net, y_net, y_evn, y_odd) -> torch.Tensor:
        ratio = None
        if self.noise > 0.0:
            # Reassembled first, so the per-shard slicing indexes global angles.
            ratio = self._noise_ratio(as_sinogram(y_evn), as_sinogram(y_odd))

        # One rotation for both halves.
        k = self._transform.get_params(x_net)["k_idx"]
        x_rot = self._transform.transform(x_net, k_idx=k)
        y_rot = self._transform.transform(y_net, k_idx=k)

        # Independent draws per half: shared eps would correlate each term's
        # input with the other half's target.
        pe, po = pair.physics_evn, pair.physics_odd
        loss = self._mse(self._recon(x_rot, pe, model, ratio), y_rot) + self._mse(
            self._recon(y_rot, po, model, ratio), x_rot
        )
        return self.weight * loss
