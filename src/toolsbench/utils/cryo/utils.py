"""Small helpers shared by the cryo solvers."""

from __future__ import annotations

import torch

__all__ = [
    "AMP_DTYPES",
    "AmpDenoiser",
    "normalize_num_operators",
    "reduce_metrics_max",
    "wants_sharding",
]

#: None means off: no autocast, no scaler, no cast anywhere.
AMP_DTYPES = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class AmpDenoiser(torch.nn.Module):
    """Run ``inner`` under autocast and return fp32.

    An ``nn.Module`` rather than a closure so it can be assigned to
    ``PGD.prior[0].denoiser``, a child-module slot. The ``.float()`` is
    load-bearing: astra has no dtype guard and ``GpuFSC``'s ``torch.fft``
    raises on bfloat16.
    """

    def __init__(self, inner, device_type: str, dtype) -> None:
        super().__init__()
        self.inner = inner
        self.device_type = device_type
        self.dtype = dtype

    def forward(self, x, *args, **kwargs):
        with torch.amp.autocast(self.device_type, dtype=self.dtype):
            return self.inner(x, *args, **kwargs).float()


def normalize_num_operators(num_operators: int | str | None) -> int | str | None:
    """Normalise the config spellings of ``num_operators`` to ``None``/``"auto"``/int.

    A YAML or CLI config hands strings through, so ``"null"``/``"none"`` are the
    unsharded case spelled out. One parser, used by both
    :func:`wants_sharding` and ``physics.resolve_num_operators``.
    """
    if not isinstance(num_operators, str):
        return num_operators
    key = num_operators.strip().lower()
    if key in ("none", "null"):
        return None
    if key == "auto":
        return "auto"
    raise ValueError(
        f"num_operators must be None, 'auto' or an int, got {num_operators!r}."
    )


def wants_sharding(num_operators: int | str | None) -> bool:
    """Whether ``num_operators`` asks for a sharded operator at all."""
    return normalize_num_operators(num_operators) is not None


def reduce_metrics_max(metrics: dict, ctx) -> dict:
    """Max every profiler metric over the ranks.

    An iteration ends when the slowest rank finishes. Done here rather than in
    ``CustomProfiler``, which is shared with solvers where rank-local timings
    are the established meaning of the column. Best-effort: a failed collective
    leaves the local values rather than failing a run over a metric.
    """
    if not metrics or ctx is None:
        return metrics
    if int(getattr(ctx, "global_world_size", 0) or 0) <= 1:
        return metrics
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return metrics
        keys = sorted(metrics)
        values = torch.tensor(
            [float(metrics[k]) for k in keys], dtype=torch.float64, device=ctx.device
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        return dict(zip(keys, values.tolist()))
    except Exception:  # pragma: no cover - a metric must not fail a run
        return metrics
