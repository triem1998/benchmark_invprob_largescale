"""Solver utilities shared across PnP solvers.

This module provides helpers for step size computation, reconstruction
initialization, normalization strategies, and training curve plotting,
used by both single-GPU and distributed PnP solvers.
"""

import torch
import torch.distributed as dist
import copy
from deepinv.utils import TensorList


def measurement_to_device(measurement, device: torch.device):
    """Move measurements to device, handling Tensor, list, and TensorList."""
    if isinstance(measurement, (list, TensorList)):
        return TensorList([m.to(device) for m in measurement])
    return measurement.to(device)


def initialize_reconstruction(
    signal_shape: tuple,
    operator,
    measurements,
    device: torch.device,
    method: str = "adjoint",
    clip_range: tuple = None,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Initialize the reconstruction signal.

    Parameters
    ----------
    signal_shape : tuple
        Shape of the signal to initialize.
    operator : deepinv.physics.Physics
        Physics operator (can be stacked or distributed).
    measurements : torch.Tensor or TensorList
        Observed measurements.
    device : torch.device
        Device to create the tensor on.
    method : str, optional
        Initialization method:
        - ``"zeros"``: start from zero (always safe; works for any physics).
        - ``"pseudo_inverse"``: ``x_0 = A†y`` clamped to ``[0, 1]`` (natural
          images / bounded domains).
        - ``"adjoint"``: ``x_0 = Aᵀy`` without clamping (radio, tomography,
          or any unbounded physical domain).
    clip_range: (sig_min, sig_max) for scaling the dirty image
    weights: Optional density weights for weighted dirty image

    Returns
    -------
    torch.Tensor
        Initialized reconstruction tensor on ``device``.
    """
    if method == "zeros":
        return torch.zeros(signal_shape, device=device)

    elif method == "pseudo_inverse":
        if weights is not None:
            # Create a temporary weighted operator for a sharper init
            weighted_op = copy.deepcopy(operator)
            weighted_op.setWeight(weights.to(device))
            dirty = weighted_op.A_dagger(measurements)
        else:
            dirty = operator.A_dagger(measurements)

        if clip_range is not None:
            sig_min, sig_max = clip_range
            # Peak-normalize dirty image to signal range and clamp
            x_init = dirty * sig_max / dirty.max()
            x_init = x_init.clamp(sig_min, sig_max)
        return x_init

    elif method == "adjoint":
        if weights is not None:
            # Create a temporary weighted operator for a sharper init
            weighted_op = copy.deepcopy(operator)
            weighted_op.setWeight(weights.to(device))
            dirty = weighted_op.A_adjoint(measurements)
        else:
            dirty = operator.A_adjoint(measurements)

        if clip_range is not None:
            sig_min, sig_max = clip_range
            # Peak-normalize dirty image to signal range and clamp
            x_init = dirty * sig_max / dirty.max()
            x_init = x_init.clamp(sig_min, sig_max)
        return x_init

    else:
        raise ValueError(
            f"Unknown initialization method: '{method}'. "
            "Choose from 'zeros', 'pseudo_inverse', or 'adjoint'."
        )


def build_solver_name(
    name_prefix: str,
    slurm_nodes: int,
    slurm_ntasks_per_node: int,
    torchrun_nproc_per_node: int,
    distributed_mode: bool,
) -> str:
    """Build a unique solver run name with timestamp and parallelism suffix."""
    import os
    from datetime import datetime

    ts = datetime.now().strftime("_%Y%m%d_%H%M%S_")
    if slurm_ntasks_per_node > 1:
        name = name_prefix + ts + f"{slurm_nodes}n{slurm_ntasks_per_node}t"
    elif torchrun_nproc_per_node > 1:
        name = name_prefix + ts + f"torchrun_{torchrun_nproc_per_node}proc"
    else:
        name = name_prefix + ts + "_single"
    if distributed_mode:
        name = name + f"_rank{int(os.environ.get('RANK', 0))}"
    return name


def get_device_from_context(ctx) -> torch.device:
    """Return ctx.device if a distributed context is provided, otherwise auto-detect."""
    if ctx is not None:
        return ctx.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_and_barrier(device: torch.device, ctx) -> None:
    """Synchronize CUDA ops and join *every* rank before the next iteration.

    ``ctx.barrier()`` defaults to the inner group. That is enough while all
    ranks cooperate on one volume, but with data-parallel replicas it would let
    them drift apart across iterations and make the per-iteration timings
    incomparable, so the global group is requested explicitly.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if ctx is None:
        return
    group = None
    if int(getattr(ctx, "dp_world_size", 1)) > 1:
        group = dist.group.WORLD
    ctx.barrier(group=group)


def distributed_callback_iter(cb, distributed_mode: bool, device: torch.device, ctx):
    """Yield while the benchopt callback returns True, broadcasting the decision in distributed mode."""
    while True:
        keep_going = cb()
        if distributed_mode and ctx is not None:
            decision = torch.tensor([float(keep_going)], device=device)
            ctx.broadcast(decision, src=0)
            keep_going = bool(decision.item())
        if not keep_going:
            return
        yield


def setup_distributed_env() -> int:
    """Initialise the distributed environment and return ``world_size``.

    Checks whether ``RANK`` / ``WORLD_SIZE`` env-vars are already set (e.g.
    launched via ``torchrun``).  If not, attempts to export them via
    ``submitit.helpers.TorchDistributedEnvironment`` (SLURM jobs).  Falls
    back silently to ``world_size=1`` for single-process runs.

    Returns
    -------
    int
        The ``WORLD_SIZE`` discovered (1 if non-distributed).
    """
    import os

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Distributed environment already initialized: world_size={world_size}")
        return world_size

    try:
        import submitit

        submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(
            f"Initialized distributed environment via submitit: world_size={world_size}"
        )
        return world_size
    except (ImportError, RuntimeError) as e:
        print(f"Running in non-distributed mode: {e}")
        return 1


def profile_roofline(model, x, sigma, bytes_per_elem=4):
    """FLOPs, memory traffic and arithmetic intensity of one forward pass.

    Counts conv/linear FLOPs and the bytes touched by every leaf module
    (inputs, outputs, weights) through forward hooks, so the numbers describe
    the model workload itself, independent of how it is executed.
    """
    total_flops, total_bytes = [0], [0]

    def hook(module, inp, out):
        for t in inp:
            if isinstance(t, torch.Tensor):
                total_bytes[0] += t.numel() * bytes_per_elem
        if isinstance(out, torch.Tensor):
            total_bytes[0] += out.numel() * bytes_per_elem
        weight = getattr(module, "weight", None)
        if weight is not None:
            total_bytes[0] += weight.numel() * bytes_per_elem

        conv_types = (
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        )
        if isinstance(module, conv_types) and isinstance(out, torch.Tensor):
            weight = module.weight
            out_spatial = out.numel() // (out.shape[0] * out.shape[1])
            kernel_ops = int(torch.tensor(weight.shape[1:]).prod().item())
            batch = inp[0].shape[0] if isinstance(inp[0], torch.Tensor) else 1
            total_flops[0] += 2 * kernel_ops * weight.shape[0] * out_spatial * batch
        elif isinstance(module, torch.nn.Linear) and isinstance(inp[0], torch.Tensor):
            batch = int(torch.tensor(inp[0].shape[:-1]).prod().item())
            total_flops[0] += 2 * module.in_features * module.out_features * batch

    hooks = [
        m.register_forward_hook(hook) for m in model.modules() if not list(m.children())
    ]
    try:
        with torch.no_grad():
            model(x, sigma=sigma)
    finally:
        for h in hooks:
            h.remove()

    flops, mem_bytes = total_flops[0], total_bytes[0]
    return dict(
        flops=flops,
        mem_bytes=mem_bytes,
        arith_intensity=flops / mem_bytes if mem_bytes > 0 else 0.0,
    )


def clamp_stepsize(model, eps: float = 1e-8) -> None:
    """Keep a learned PGD stepsize positive after each optimizer step.

    No-op when the stepsize is not trainable, so it is safe to call on any
    unfolded model — ``solver/equivariant.py`` and ``solver/unrolled_pnp.py`` both
    build a PGD whose ``stepsize`` can be in ``trainable_params``.
    """
    stepsize = model.params_algo["stepsize"]
    if isinstance(stepsize, torch.nn.ParameterList):
        with torch.no_grad():
            for s in stepsize:
                s.data.clamp_(min=eps)
