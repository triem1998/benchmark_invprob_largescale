"""Solver utilities shared across PnP solvers.

This module provides helpers for step size computation, reconstruction
initialization, normalization strategies, and training curve plotting,
used by both single-GPU and distributed PnP solvers.
"""

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _to_mid_slice(x):
    """Return the central 2-D slice from the first sample of a 3-D or batched tensor."""
    t = x.detach().float()
    if t.ndim == 5:
        t = t[0, 0]
    elif t.ndim == 4:
        t = t[0, 0]
    if t.ndim == 3:
        t = t[t.shape[0] // 2]
    return t.cpu().numpy()


def save_training_figure(
    out_path,
    x,
    x_sparse,
    x_hat,
    psnr_db: float,
    title: str,
    x_init=None,
    psnr_init_db=None,
    psnr_sparse_db=None,
) -> None:
    """Save a central-slice comparison figure (GT | Sparse | Init | Pred).

    Works for any 2-D or 3-D tensor: a single central slice is extracted and
    displayed per panel.  Suitable for both natural-image (Urban100) and
    volumetric (Walnut CT) training runs.

    Parameters
    ----------
    out_path : str or Path
        Destination file path (parent directories are created automatically).
    x : Tensor
        Ground-truth image/volume.
    x_sparse : Tensor or None
        Sparse/FBP warm-start reconstruction, or ``None`` to omit the panel.
    x_hat : Tensor
        Model prediction.
    psnr_db : float
        PSNR of *x_hat* vs *x* (shown in the panel title).
    title : str
        Figure suptitle (e.g. ``"epoch=5 step=12"``).
    x_init : Tensor or None
        Untrained-model prediction, or ``None`` to omit the panel.
    psnr_init_db : float or None
        PSNR of *x_init* (shown when provided).
    psnr_sparse_db : float or None
        PSNR of *x_sparse* (shown when provided).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    gt = _to_mid_slice(x)
    pred = _to_mid_slice(x_hat)
    sparse = _to_mid_slice(x_sparse) if x_sparse is not None else None
    init_img = _to_mid_slice(x_init) if x_init is not None else None

    sparse_label = (
        f"Sparse ({psnr_sparse_db:.2f} dB)" if psnr_sparse_db is not None else "Sparse"
    )
    init_label = (
        f"Untrained ({psnr_init_db:.2f} dB)"
        if psnr_init_db is not None
        else "Untrained"
    )
    panels = [("GT", gt)]
    if sparse is not None:
        panels.append((sparse_label, sparse))
    if init_img is not None:
        panels.append((init_label, init_img))
    panels.append((f"Pred ({psnr_db:.2f} dB)", pred))

    vmin, vmax = float(gt.min()), float(gt.max())
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_prediction_results(tensor, path) -> None:
    """Save the first 3-D volume from a (possibly batched) tensor to a .pt file."""
    import torch

    t = tensor.detach().cpu()
    if t.ndim == 5:
        t = t[0]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, p)


def crop_psnr(x_hat, x, crop: int = 100):
    """PSNR over the central crop of a 3-D volume, normalised by the crop's range.

    Avoids cone-beam edge artefacts.  Both tensors must have shape (B, C, D, H, W).

    Returns a scalar Tensor (dB).
    """
    import torch

    c = slice(crop, -crop)
    tgt = x[..., c, c, c]
    pred = x_hat[..., c, c, c]
    range_val = tgt.amax() - tgt.amin()
    mse = torch.mean((tgt - pred) ** 2)
    return 10.0 * torch.log10((range_val**2).clamp(min=1e-12) / mse.clamp(min=1e-12))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
