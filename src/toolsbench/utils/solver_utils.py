"""Solver utilities shared across PnP solvers.

This module provides helpers for step size computation, reconstruction
initialization, and normalization strategies, used by both single-GPU
and distributed PnP solvers.
"""

import torch


def compute_step_size_from_operator(operator, ground_truth: torch.Tensor) -> float:
    """Compute PnP step size as 1 / Lipschitz constant of the forward operator.

    Parameters
    ----------
    operator : deepinv.physics.Physics
        Physics operator (can be stacked or distributed).
    ground_truth : torch.Tensor
        Ground truth tensor used to create an example signal for norm computation.

    Returns
    -------
    float
        Step size = 1 / lipschitz_constant, or 1.0 if constant is non-positive.
    """
    with torch.no_grad():
        x_example = torch.zeros_like(
            ground_truth, device=ground_truth.device, dtype=ground_truth.dtype
        )
        lipschitz_constant = operator.compute_norm(x_example, local_only=False)
        return 1.0 / lipschitz_constant if lipschitz_constant > 0 else 1.0


def initialize_reconstruction(
    signal_shape: tuple,
    operator,
    measurements,
    device: torch.device,
    method: str = "pseudo_inverse",
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

    Returns
    -------
    torch.Tensor
        Initialized reconstruction tensor on ``device``.
    """
    if method == "zeros":
        return torch.zeros(signal_shape, device=device)

    elif method == "pseudo_inverse":
        x_init = operator.A_dagger(measurements)
        return x_init.clamp(0, 1)

    elif method == "adjoint":
        return operator.A_adjoint(measurements)

    else:
        raise ValueError(
            f"Unknown initialization method: '{method}'. "
            "Choose from 'zeros', 'pseudo_inverse', or 'adjoint'."
        )


def normalize_to_unit(x: torch.Tensor, eps: float = 1e-10):
    """Normalize a tensor to [0, 1] using its current min/max.

    Returns the normalized tensor together with the offset and scale so the
    mapping can be inverted with :func:`denormalize_from_unit`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (any shape).
    eps : float, optional
        Minimum scale to avoid division by zero. Default: 1e-10.

    Returns
    -------
    x_01 : torch.Tensor
        Tensor normalized to [0, 1].
    norm_min : float
        Minimum value used for normalization.
    norm_scale : float
        Scale = max - min (clamped to ``eps``).
    """
    norm_min = x.min().item()
    norm_scale = max((x.max() - x.min()).item(), eps)
    x_01 = (x - norm_min) / norm_scale
    return x_01, norm_min, norm_scale


def denormalize_from_unit(
    x_01: torch.Tensor, norm_min: float, norm_scale: float
) -> torch.Tensor:
    """Inverse of :func:`normalize_to_unit`.

    Maps a tensor from [0, 1] back to the original physical domain using
    the offset and scale returned by :func:`normalize_to_unit`.

    Parameters
    ----------
    x_01 : torch.Tensor
        Tensor in [0, 1].
    norm_min : float
        Offset (minimum of original tensor).
    norm_scale : float
        Scale (max - min of original tensor).

    Returns
    -------
    torch.Tensor
        Tensor in the original physical domain.
    """
    return x_01 * norm_scale + norm_min
