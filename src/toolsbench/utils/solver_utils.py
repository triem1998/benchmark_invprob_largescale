"""Solver utilities shared across PnP solvers.

This module provides helpers for step size computation, reconstruction
initialization, normalization strategies, and training curve plotting,
used by both single-GPU and distributed PnP solvers.
"""

from pathlib import Path

import torch


def save_training_curves(
    loss_steps: list,
    loss_epochs: list,
    train_psnr: list,
    val_psnr: list,
    save_dir,
    prefix: str = "",
    grad_accumulation_steps: int = 1,
) -> None:
    """Save loss and PSNR training curves as PNG figures.

    Creates two files in ``save_dir``:
      - ``{prefix}loss.png``  — MSE loss per gradient step (thin), optionally
        averaged over ``grad_accumulation_steps`` to show per-optimizer-step
        loss (medium), and per-epoch average (thick).
      - ``{prefix}psnr.png``  — train PSNR and val PSNR per epoch.

    Parameters
    ----------
    loss_steps : list of float
        Per-gradient-step MSE loss values.
    loss_epochs : list of float
        Per-epoch mean MSE loss values.
    train_psnr : list of float
        Per-epoch mean train PSNR (dB).
    val_psnr : list of float
        Per-epoch mean validation PSNR (dB).
    save_dir : str or Path
        Directory where the figures are written (created if needed).
    prefix : str, optional
        Optional filename prefix, e.g. ``"rank0_"``.
    grad_accumulation_steps : int, optional
        Number of forward passes accumulated before each optimizer step.
        When > 1, an additional per-optimizer-step averaged loss line is shown.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend, safe on clusters
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_training_curves] matplotlib not available — skipping plots")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: Loss
    # ------------------------------------------------------------------
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    accum = max(1, int(grad_accumulation_steps))
    if loss_steps:
        ax_loss.plot(
            range(len(loss_steps)),
            loss_steps,
            color="steelblue",
            alpha=0.3,
            linewidth=0.7,
            label="loss (per step)",
        )
    # Per-optimizer-step average (only meaningful when accum > 1)
    if loss_steps and accum > 1:
        n = len(loss_steps)
        # Pad to a multiple of accum, then reshape and average
        pad = (-n) % accum
        padded = list(loss_steps) + [loss_steps[-1]] * pad
        opt_x = [accum * (i + 1) - 1 for i in range(len(padded) // accum)]
        opt_y = [
            sum(padded[i * accum:(i + 1) * accum]) / accum
            for i in range(len(padded) // accum)
        ]
        # Clip to valid range (no padded x beyond data length)
        opt_x = [x for x in opt_x if x < n]
        opt_y = opt_y[:len(opt_x)]
        ax_loss.plot(
            opt_x,
            opt_y,
            color="darkorange",
            alpha=0.8,
            linewidth=1.4,
            marker=".",
            markersize=3,
            label=f"loss (per optimizer step, accum={accum})",
        )
    if loss_epochs:
        # Mark epoch boundaries at the last step of each epoch (0-indexed)
        steps_per_epoch = (
            len(loss_steps) // len(loss_epochs) if loss_steps and loss_epochs else 1
        )
        epoch_x = [(i + 1) * steps_per_epoch - 1 for i in range(len(loss_epochs))]
        ax_loss.plot(
            epoch_x,
            loss_epochs,
            color="steelblue",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="loss (epoch mean)",
        )
    ax_loss.set_xlabel("Gradient step")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    fig_loss.tight_layout()
    loss_path = save_dir / f"{prefix}loss.png"
    fig_loss.savefig(loss_path, dpi=120)
    plt.close(fig_loss)

    # ------------------------------------------------------------------
    # Figure 2: PSNR
    # ------------------------------------------------------------------
    fig_psnr, ax_psnr = plt.subplots(figsize=(8, 4))
    epochs = range(len(train_psnr))
    if train_psnr:
        ax_psnr.plot(
            epochs,
            train_psnr,
            color="royalblue",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="train PSNR",
        )
    if val_psnr:
        ax_psnr.plot(
            range(len(val_psnr)),
            val_psnr,
            color="tomato",
            linewidth=1.8,
            marker="s",
            markersize=4,
            label="val PSNR",
        )
    ax_psnr.set_xlabel("Epoch")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("PSNR — train vs validation")
    ax_psnr.legend()
    ax_psnr.grid(True, linestyle="--", alpha=0.5)
    fig_psnr.tight_layout()
    psnr_path = save_dir / f"{prefix}psnr.png"
    fig_psnr.savefig(psnr_path, dpi=120)
    plt.close(fig_psnr)

    print(
        f"[save_training_curves] saved → {loss_path.name}, {psnr_path.name}",
        flush=True,
    )


def save_reconstruction_figure(
    x,
    x_sparse,
    x_hat,
    psnr_val: float,
    epoch: int,
    save_dir,
    prefix: str = "",
    x_hat_init=None,
    psnr_init=None,
) -> None:
    """Save a reconstruction comparison figure for one validation sample.

    Detects 2D (B,C,H,W) vs 3D (B,C,D,H,W) automatically.

    Column order (left → right):
      Ground Truth | [Sparse Recon PSNR=xx] | [Unrolled (init) PSNR=xx] | Output PSNR=xx

    Optional columns are omitted when the corresponding tensor is None.
    All panels in a row share the same ``vmin``/``vmax`` (taken from the
    ground-truth slice).

    Saved as ``{save_dir}/{prefix}recon_epoch{epoch:04d}.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_reconstruction_figure] matplotlib not available — skipping")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Move to CPU numpy, take first sample first channel.
    def to_np(t):
        if t is None:
            return None
        return t[0, 0].detach().cpu().float().numpy()

    import numpy as np

    gt   = to_np(x)
    sp   = to_np(x_sparse)    # may be None
    init = to_np(x_hat_init)  # unrolled with initial weights, may be None
    out  = to_np(x_hat)

    # Compute PSNR of an array vs ground truth (crop centre 100px for 3D).
    def _arr_psnr(gt_arr, arr, crop: int = 100):
        if arr is None:
            return None
        if gt_arr.ndim == 3 and min(gt_arr.shape) > 2 * crop:
            c = slice(crop, -crop)
            g, s = gt_arr[c, c, c], arr[c, c, c]
        else:
            g, s = gt_arr, arr
        range_val = g.max() - g.min()
        if range_val < 1e-12:
            return None
        mse = float(np.mean((g - s) ** 2))
        if mse < 1e-12:
            return float("inf")
        return 10.0 * np.log10(range_val ** 2 / mse)

    sparse_psnr = _arr_psnr(gt, sp)
    # Use caller-supplied init PSNR (computed over full val set) if available,
    # else fall back to per-image estimate.
    if psnr_init is None:
        psnr_init = _arr_psnr(gt, init)

    # Build ordered column list: (array, label)
    columns = [("GT", gt, "Ground Truth")]
    if sp is not None:
        sp_label = (
            f"Sparse  PSNR={sparse_psnr:.2f} dB" if sparse_psnr is not None else "Sparse Recon"
        )
        columns.append(("SP", sp, sp_label))
    if init is not None:
        init_label = (
            f"Unrolled (init)  PSNR={psnr_init:.2f} dB"
            if psnr_init is not None else "Unrolled (init)"
        )
        columns.append(("INIT", init, init_label))
    out_label = f"Trained  PSNR={psnr_val:.2f} dB"
    columns.append(("OUT", out, out_label))

    n_cols = len(columns)
    is_3d = gt.ndim == 3  # (D, H, W)

    if not is_3d:
        # ---- 2D ----
        vmin, vmax = float(gt.min()), float(gt.max())
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        for ax, (_, img, title) in zip(axes, columns):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch}  —  validation sample", fontsize=10)
        fig.tight_layout()
    else:
        # ---- 3D: axial / coronal / sagittal ----
        D, H, W = gt.shape
        row_labels = [f"Axial  (z={D//2})", f"Coronal  (y={H//2})", f"Sagittal  (x={W//2})"]

        def _slices(vol):
            if vol is None:
                return [None, None, None]
            return [vol[D//2, :, :], vol[:, H//2, :], vol[:, :, W//2]]

        # slices_per_col: list of (key, [3 slices], label)
        slices_per_col = [(key, _slices(arr), lbl) for key, arr, lbl in columns]

        fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
        # Ensure axes is always 2-D
        if n_cols == 1:
            axes = axes.reshape(3, 1)
        gt_slices = _slices(gt)
        for row in range(3):
            vmin = float(gt_slices[row].min())
            vmax = float(gt_slices[row].max())
            for col_idx, (_, sl_list, clabel) in enumerate(slices_per_col):
                ax = axes[row, col_idx]
                img = sl_list[row]
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                else:
                    ax.set_visible(False)
                if row == 0:
                    ax.set_title(clabel, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8)
                ax.axis("off")
        fig.suptitle(f"Epoch {epoch}  —  validation sample", fontsize=10)
        fig.tight_layout()

    out_path = save_dir / f"{prefix}recon_epoch{epoch:04d}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[save_reconstruction_figure] saved → {out_path.name}", flush=True)


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
