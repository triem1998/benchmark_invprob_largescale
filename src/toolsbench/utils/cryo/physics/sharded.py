"""Angle-sharded tomography physics.
The backend registry, the angle split and the sharded operator are carried over verbatim.

"""

from __future__ import annotations

import importlib.util

from functools import partial

import torch

# deepinv has moved this between layouts: re-exported from the package on some
# revisions, only from the ``framework`` subpackage on others.
try:
    from deepinv.distributed import DistributedStackedLinearPhysics
except ImportError:  # pragma: no cover - depends on the installed deepinv
    from deepinv.distributed.framework import DistributedStackedLinearPhysics
from deepinv.utils.tensorlist import TensorList

from .tomography import TomographyEM
from .tomography_torch import TomographyEMTorch

#: The interchangeable operators, keyed by the ``tomography_backend`` config
#: value. They share a constructor signature, so the backend is a pure lookup.
TOMOGRAPHY_BACKENDS = {
    "astra": TomographyEM,
    # astra's back-projector is not a true transpose, so reproducing it keeps the
    # gradient unchanged when ``auto`` flips astra -> torch; ~4x quicker too.
    "torch": partial(TomographyEMTorch, adjoint_mode="fast"),
    "torch_exact": TomographyEMTorch,  # true transpose: a real PGD gradient
}


def resolve_tomography_backend(backend: str, device) -> str:
    """Turn ``cfg.tomography_backend`` into a concrete key of ``TOMOGRAPHY_BACKENDS``.

    ``"astra"``/``"torch"`` are taken as given (so a run can be forced onto
    either). ``"auto"`` picks astra only where it can actually run — astra ships
    CUDA kernels, so a ROCm build of torch or a CPU device must fall back to the
    pure-torch operator, which is numerically equivalent (see
    ``tests/test_tomography_torch.py``) at ~4-5x the cost.
    """
    if backend not in ("auto", *TOMOGRAPHY_BACKENDS):
        raise ValueError(
            f"tomography_backend must be 'auto' or one of "
            f"{sorted(TOMOGRAPHY_BACKENDS)}, got {backend!r}."
        )
    if backend != "auto":
        return backend
    astra_usable = (
        torch.device(device).type == "cuda"
        and torch.version.hip is None
        and importlib.util.find_spec("astra") is not None
    )
    return "astra" if astra_usable else "torch"


def projection_splits(num_angles: int, num_operators: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` angle ranges, one per operator."""
    base, rem = divmod(int(num_angles), int(num_operators))
    sizes = [base + (1 if i < rem else 0) for i in range(num_operators)]
    edges = [0]
    for s in sizes:
        edges.append(edges[-1] + s)
    return [(edges[i], edges[i + 1]) for i in range(num_operators)]


def split_sinogram(y: torch.Tensor, num_operators: int) -> TensorList:
    """Split a ``(B, C, V, A, N)`` sinogram along the angle axis to match the
    sharded operators — the measurement counterpart of ``projection_splits``.
    """
    chunks = projection_splits(int(y.shape[3]), num_operators)
    return TensorList([y[:, :, :, s:e, :].contiguous() for (s, e) in chunks])


class ShardedTomography(DistributedStackedLinearPhysics):
    """Angle-sharded tomography + the one method deepinv's container lacks: ``fbp``.

    ``fbp`` is a ``TomographyEM`` method, not part of ``LinearPhysics``, so the
    distributed container has none. It is the same map-reduce as ``A_adjoint``
    (each shard back-projects its own angles, the volumes are summed across
    ranks) plus the two corrections that keep it identical to the unsharded
    operator: each shard divides by its *own* angle count, so it is reweighted
    by ``A_i / n_angles_total`` (attached when the shards are built); and the DC
    centring uses the global sinogram mean, since centring is shift-idempotent
    and so cannot be recovered shard by shard.
    """

    def fbp(self, y, gather: bool = True, reduce_op: str | None = "sum", **kwargs):
        if len(y) != self.num_operators:
            raise ValueError(
                f"fbp needs the whole sinogram (all {self.num_operators} pieces, as "
                f"returned by A(x)), got {len(y)}: the global DC mean cannot be formed "
                f"from a subset, and centring per shard is not equivalent."
            )
        # Every rank holds every piece (A gathers), so the global mean is local
        # arithmetic — no collective. Summing first and dividing once is the plain
        # definition of the mean; the A_i/A reweighting below is still needed
        # because each shard's fbp_raw divides by its *own* angle count.
        count = sum(t.shape[-3] * t.shape[-2] * t.shape[-1] for t in y)
        mean = sum(t.sum(dim=(-3, -2, -1), keepdim=True) for t in y) / count
        return self._map_reduce_gather(
            [y[i] - mean for i in self.local_indexes],
            lambda p, t, **kw: p.fbp_raw(t) * (p.n_angles / self.n_angles_total),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )


def measure_opnorm_sq(physics, init: torch.Tensor) -> float:
    """``||A^T A||_2`` of a distributed (sharded) operator.

    Only needed when sharding: the shards are built with ``normalize=False``
    because each one's own norm is not the full operator's.

    ``local_only=False`` runs the power iteration over the *assembled* operator,
    communicating at each step. deepinv's default (``True``) only sums the
    per-shard norms, an upper bound that grows with the shard count — which
    would make the stepsize, and so the reconstruction, depend on
    ``num_operators``. Paid once at build time, not per step.
    """
    # Full (B, C, D, H, W) init, not the unbatched form deepinv's docstring
    # suggests: the power iteration feeds x0 straight into A, and astra's
    # forward unpacks five dims.
    return float(physics.compute_sqnorm(init, local_only=False, verbose=False))


def normalize_sharded(physics, init: torch.Tensor) -> float:
    """Give a sharded operator the unit spectral norm ``normalize=True`` gives
    the unsharded one, by rescaling every shard with the *global* norm.

    Scaling only the PGD stepsize by :math:`1/\\|A\\|^2` fixes the :math:`A^{T}A` term of the
    data-fidelity gradient but leaves the :math:`A^{T}y` term off by one factor of the norm,
    so the two paths converge to different reconstructions. Normalising the
    operator itself makes the sharded and unsharded physics identical.

    :return: the measured ``||A^T A||_2`` before normalisation (diagnostic).
    """
    sqnorm = measure_opnorm_sq(physics, init)
    for p in physics.local_physics:
        # astra holds the two knobs on its wrapper, the torch operator on itself.
        target = getattr(p, "xray", p)
        target.operator_norm = sqnorm**0.5
        target.normalize = True
    return sqnorm
