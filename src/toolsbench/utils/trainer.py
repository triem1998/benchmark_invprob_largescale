"""Generic per-epoch / per-batch trainer for unrolled inverse-problem solvers.

The :class:`_Trainer` is decoupled from any specific benchopt ``Solver`` so
that it can be imported and reused across multiple solver files.

Supported batch formats
-----------------------
**Dict format** (Walnut CT / tomography pipeline)::

    {
        "x":             ground-truth volume  (B, C, D, H, W),
        "x_sparse":      FBP warm-start       (B, C, D, H, W) | None,
        "y_full":        full-angle sinogram   (B, 1, n_angles, det),
        "num_proj":      number of projections used (int),
        "operator_norm": per-sample Lipschitz constant (float),
    }

**Tuple format** (Urban100 / generic 2-D / any other dataset)::

    (x, y_tl)                             — minimal 2-item
    (x, x_sparse, y_tl, num_proj_key)     — with sparse init + multi-physics key
    (x, x_sparse, y_tl, num_proj_key, opnorm) — full 5-item

The dispatch between tomo-specific helpers (sinogram splitting, crop-PSNR,
volume saving) and generic helpers (standard PSNR) is driven entirely by
``num_proj_key``: when it is ``None`` the generic path is taken.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from .tomo_utils import (
    append_metrics_row,
    ensure_dir,
    split_sinogram,
)
from .solver_utils import (
    crop_psnr,
    normalize_to_unit,
    save_prediction_results,
    save_training_figure,
)


@dataclass
class TrainingHistory:
    """Container for all per-epoch and per-step training/validation metrics.

    All list fields are populated in-place during training so that external
    references (e.g. shared with :class:`_Trainer`) remain valid.
    """

    # Per-epoch PSNR
    train_psnr: list[float] = field(default_factory=list)
    val_psnr: list[float] = field(default_factory=list)
    # Per-step loss (shared references to _Trainer.all_loss_steps / all_val_loss_steps)
    loss_steps: list[float] = field(default_factory=list)
    val_loss_steps: list[float] = field(default_factory=list)
    # Per-epoch mean loss
    loss_epochs: list[float] = field(default_factory=list)
    val_loss_epochs: list[float] = field(default_factory=list)
    # Per-epoch GPU memory snapshots (dicts with allocated/reserved/available keys)
    gpu_metrics: list[dict[str, Any]] = field(default_factory=list)


class _Trainer:
    """Per-batch training / evaluation logic, decoupled from benchopt's Solver.

    Parameters
    ----------
    model:
        The unrolled model (e.g. PGD with ``unfold=True``).
    optimizer:
        PyTorch optimiser that owns all trainable parameters.
    physics_map:
        Mapping ``{num_proj_key: physics_object}``.  Use ``{None: physics}``
        for single-physics datasets.
    num_operators_map:
        Mapping ``{num_proj_key: n_operators}`` — used for sinogram splitting
        in the dict (Walnut CT) batch format.
    device:
        Torch device to move tensors to.
    output_dir:
        Root directory for CSV metrics, debug images, and volume snapshots.
    min_pixel:
        Lower clip value applied to ``x_hat`` before PSNR computation (e.g.
        0.0 to suppress negative attenuation artefacts).
    ctx:
        Optional distributed context providing ``.rank``.  When ``None`` the
        trainer behaves as rank-0 single-process.
    grad_accumulation_steps:
        Number of forward passes before an optimiser step.
    clip_grad_norm:
        Maximum gradient L2 norm passed to
        ``torch.nn.utils.clip_grad_norm_``.
    max_batches_per_epoch:
        Cap on the number of training batches per epoch (``None`` = no cap).
    log_every:
        Print a progress line every *log_every* steps.
    save_debug_every:
        Save a debug image every *save_debug_every* steps (0 = disabled).
    use_x_init:
        Pass ``x_sparse`` as warm-start ``init`` to the model when available.
    normalize:
        Normalise ``x`` (and ``x_sparse``) to ``[0, 1]`` per-batch before the
        forward pass.
    train_algo_params:
        When ``True`` the optimiser is expected to include algorithmic
        parameters (stepsize, g_param, optionally beta).  Used for the
        post-step parameter clamping.
    lambda_relaxation:
        When ``True`` clamp ``beta`` parameters to ``[0, 1]`` after each
        optimiser step.
    ref_opnorm:
        Fallback operator norm used for tuple-format batches that do not
        carry a per-sample norm.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        physics_map: dict,
        num_operators_map: dict,
        device,
        output_dir: "Path",
        min_pixel: float = 0.0,
        ctx=None,
        grad_accumulation_steps: int = 1,
        clip_grad_norm: float = 1.0,
        max_batches_per_epoch=None,
        log_every: int = 1,
        save_debug_every: int = 0,
        use_x_init: bool = True,
        normalize: bool = False,
        train_algo_params: bool = True,
        lambda_relaxation: bool = False,
        ref_opnorm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.physics_map = physics_map
        self.num_operators_map = num_operators_map
        self.device = device
        self.min_pixel = float(min_pixel)
        self.ctx = ctx
        self.grad_accumulation_steps = max(1, int(grad_accumulation_steps))
        self.clip_grad_norm = float(clip_grad_norm)
        self.max_batches_per_epoch = max_batches_per_epoch
        self.log_every = int(log_every)
        self.save_debug_every = int(save_debug_every)
        self.use_x_init = use_x_init
        self.normalize = normalize
        self.train_algo_params = train_algo_params
        self.lambda_relaxation = lambda_relaxation
        self.ref_opnorm = float(ref_opnorm)

        # -----------------------------------------------------------------
        # Loss / metric history — accumulated across epochs.
        # ``all_loss_steps`` is exposed as a *shared reference* to the Solver
        # so that both objects see the same list without copying.
        # -----------------------------------------------------------------
        self.all_loss_steps: list[float] = []  # train loss per step
        self.all_val_loss_steps: list[float] = []  # val   loss per step

        # Per-epoch timing history (appended once per train_epoch call).
        # total = full epoch wall-clock; per-step values are means over all steps.
        self.epoch_train_total_times: list[float] = []  # total epoch wall-clock
        self.epoch_fwd_times: list[float] = []  # mean per-step forward time
        self.epoch_bwd_times: list[float] = []  # mean per-step backward time
        self.epoch_other_times: list[float] = (
            []
        )  # per-step: total/n - fwd - bwd (optimizer + misc)

        # Per-epoch val timing (appended once per evaluate call).
        self.epoch_val_times: list[float] = []  # total val wall-clock
        self.epoch_val_per_sample_times: list[float] = []  # total / n_samples

        # Cached reconstruction batches for visualisation (CPU tensors).
        self._cached_val_batch = None  # (x, xs, xh) from latest val run
        self._cached_init_val_batch = None  # same, captured before training

        # Per-sample init cache: keyed by (step_idx, num_proj_key).
        # Populated during the cache_as_init=True evaluate call so that
        # every subsequent val step can show a GT | Sparse | Init | Pred panel.
        self._init_images: dict = {}  # key -> x_hat_clipped (cpu tensor)
        self._init_psnrs: dict = {}  # key -> float

        # Lazy standard-PSNR metric (generic / Urban100 path).
        self._psnr_fn = None

        # Output directories — created only on rank 0 (non-rank-0 processes
        # never write to them, so there's no need to create empty dirs).
        output_dir = Path(output_dir)
        _is_rank0 = ctx is None or ctx.rank == 0
        _maybe_dir = ensure_dir if _is_rank0 else (lambda p: p)
        self._metrics_dir = _maybe_dir(output_dir / "metrics")
        self._debug_train_dir = _maybe_dir(output_dir / "debug_train")
        self._debug_val_dir = _maybe_dir(output_dir / "debug_val")
        self._predictions_dir = _maybe_dir(output_dir / "predictions")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unwrap_batch(self, batch):
        """Move *batch* to device and return a standard 5-tuple.

        Returns
        -------
        (x, x_sparse, y_tl, num_proj_key, opnorm)
        """
        from deepinv.utils.tensorlist import TensorList

        def _to(t):
            if t is None:
                return None
            if isinstance(t, (list, TensorList)):
                return TensorList([ti.to(self.device) for ti in t])
            return t.to(self.device)

        # --- dict format (Walnut CT / tomography) -----------------------
        if isinstance(batch, dict):
            x = _to(batch["x"])
            x_sparse = _to(batch.get("x_sparse"))
            num_proj = int(batch["num_proj"])
            opnorm = float(batch["operator_norm"])
            y_full = _to(batch["y_full"])
            n_ops = self.num_operators_map.get(
                num_proj, self.num_operators_map.get(None, 1)
            )
            y_tl = split_sinogram(y_full, n_ops)
            return x, x_sparse, y_tl, num_proj, opnorm

        # --- tuple format (Urban100 / generic) --------------------------
        if len(batch) == 2:
            x, y_tl = batch
            return _to(x), None, _to(y_tl), None, self.ref_opnorm
        if len(batch) == 4:
            x, x_sparse, y_tl, num_proj_key = batch
            return _to(x), _to(x_sparse), _to(y_tl), num_proj_key, self.ref_opnorm
        x, x_sparse, y_tl, num_proj_key, opnorm = batch
        return _to(x), _to(x_sparse), _to(y_tl), num_proj_key, float(opnorm)

    def _get_physics(self, num_proj_key):
        """Return the physics object for *num_proj_key* (falls back to ``None`` key)."""
        if num_proj_key in self.physics_map:
            return self.physics_map[num_proj_key]
        return self.physics_map[None]

    def _compute_psnr(self, x_hat, x, num_proj_key):
        """Dispatch to crop-PSNR (tomo) or standard PSNR (generic).

        When *num_proj_key* is not ``None`` the batch originated from a
        tomography dataset and the central-crop PSNR is used to avoid
        cone-beam edge artefacts.  Otherwise a standard mean PSNR is used.
        """
        if num_proj_key is not None:
            # Tomo path: PSNR on the central crop of the 3-D volume.
            return crop_psnr(x_hat, x)
        # Generic path: standard full-image PSNR.
        if self._psnr_fn is None:
            from deepinv.loss.metric import PSNR

            self._psnr_fn = PSNR(reduction="mean")
        return self._psnr_fn(x_hat, x)

    def _apply_stepsize_scale(self, scale: float, *, restore: bool = False):
        """Multiply (or divide) all stepsizes in-place.

        This implements the per-sample operator-norm rescaling:
          effective_step = stored_stepsize / sample_opnorm

        ``restore=False``  →  multiply by *scale*  (before forward pass)
        ``restore=True``   →  divide   by *scale*  (after forward/backward)

        Handles both trainable (``torch.Tensor``) and non-trainable (plain
        Python scalar) stepsizes.  The non-trainable case arises when
        ``train_algo_params=False`` — stepsizes are plain floats in params_algo
        and must be replaced by index.
        """
        stepsizes = self.model.params_algo["stepsize"]
        with torch.no_grad():
            for i, s in enumerate(stepsizes):
                if isinstance(s, torch.Tensor):
                    if restore:
                        s.data.div_(scale)
                    else:
                        s.data.mul_(scale)
                else:
                    # Plain Python scalar — replace in the list directly.
                    stepsizes[i] = s / scale if restore else s * scale

    def _apply_stepsize_grad_correction(self, scale: float):
        """Chain-rule correction on stepsize gradients after backward.

        ∂L/∂α_stored = ∂L/∂(α_stored · scale) · scale
        """
        with torch.no_grad():
            for s in self.model.params_algo["stepsize"]:
                if not isinstance(s, torch.Tensor):
                    continue
                if s.grad is not None:
                    s.grad.mul_(scale)

    def _optimizer_step_and_clamp(self):
        """Gradient-clip → optimiser step → zero grad → clamp algo params."""
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            max_norm=self.clip_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            for s in self.model.params_algo["stepsize"]:
                if isinstance(s, torch.Tensor):
                    s.clamp_(min=1e-8)
            if self.lambda_relaxation:
                for b in self.model.params_algo["beta"]:
                    if isinstance(b, torch.Tensor):
                        b.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader, epoch: int) -> "tuple[float, float]":
        """Run one full training epoch.

        Timing is summarised at epoch end:
        - *total*: full epoch wall-clock.
        - *per step fwd/bwd*: mean forward / backward times.
        - *per step other*: (total − Σfwd − Σbwd) / n — captures optimizer,
          gradient clipping, prefetch wait, and any other overhead.

        These values are appended to :attr:`epoch_train_total_times`,
        :attr:`epoch_fwd_times`, :attr:`epoch_bwd_times`, and
        :attr:`epoch_other_times` (rank-0 only).

        Debug figures are saved to ``debug_train/`` every
        ``max(1, save_debug_every)`` steps, independently of the log cadence.

        Returns
        -------
        (mean_psnr, mean_loss) over all steps in the epoch.
        """
        self.model.train()
        psnr_vals: list[float] = []
        loss_vals: list[float] = []
        load_times: list[float] = []
        fwd_times: list[float] = []
        bwd_times: list[float] = []
        save_times: list[float] = []
        accum_steps = self.grad_accumulation_steps
        self.optimizer.zero_grad()
        is_rank0 = self.ctx is None or self.ctx.rank == 0
        _epoch_t0 = time.perf_counter()

        loader_iter = islice(enumerate(dataloader), self.max_batches_per_epoch)
        total_steps = (
            min(len(dataloader), self.max_batches_per_epoch)
            if self.max_batches_per_epoch is not None
            else len(dataloader)
        )

        for step_idx, batch in loader_iter:
            _t0 = time.perf_counter()
            x, x_sparse, y_tl, num_proj_key, sample_opnorm = self._unwrap_batch(batch)
            _t1 = time.perf_counter()
            physics = self._get_physics(num_proj_key)

            # Optional per-batch normalisation to [0, 1].
            if self.normalize:
                x, _norm_min, _norm_scale = normalize_to_unit(x)
                if x_sparse is not None:
                    x_sparse = (x_sparse - _norm_min) / _norm_scale

            # Scale stepsizes by 1/opnorm for this batch.
            _step_scale = 1.0 / max(sample_opnorm, 1e-8)
            self._apply_stepsize_scale(_step_scale)

            _init = x_sparse if (self.use_x_init and x_sparse is not None) else None
            x_hat = self.model(y_tl, physics, init=_init)
            _t2 = time.perf_counter()

            loss = torch.nn.functional.mse_loss(x_hat, x)
            (loss / accum_steps).backward()

            # Chain-rule correction + restore stepsizes.
            self._apply_stepsize_grad_correction(_step_scale)
            self._apply_stepsize_scale(_step_scale, restore=True)
            _t3 = time.perf_counter()

            # ---- record timings / losses --------------------------------
            load_times.append(_t1 - _t0)
            fwd_times.append(_t2 - _t1)
            bwd_times.append(_t3 - _t2)
            step_loss = loss.item()
            loss_vals.append(step_loss)
            self.all_loss_steps.append(step_loss)

            # ---- compute PSNR (no grad) ---------------------------------
            with torch.no_grad():
                x_hat_clipped = x_hat.clamp(min=self.min_pixel)
                step_psnr = float(self._compute_psnr(x_hat_clipped, x, num_proj_key))
                psnr_vals.append(step_psnr)

            # ---- logging ------------------------------------------------
            if is_rank0 and (step_idx + 1) % max(1, self.log_every) == 0:
                np_key = num_proj_key if num_proj_key is not None else "-"
                print(
                    f"  [train step {step_idx + 1}/{total_steps}] "
                    f"np={np_key} | opnorm={sample_opnorm:.3f} | "
                    f"loss={step_loss:.6f} | psnr={step_psnr:.2f} dB | "
                    f"fwd={_t2-_t1:.3f}s bwd={_t3-_t2:.3f}s",
                    flush=True,
                )
                append_metrics_row(
                    self._metrics_dir / "train_steps.csv",
                    {
                        "epoch": epoch,
                        "step": step_idx,
                        "loss": step_loss,
                        "psnr": step_psnr,
                    },
                )

            # ---- debug figure (independent of log_every) ----------------
            if is_rank0 and (step_idx + 1) % max(1, self.save_debug_every) == 0:
                _ts = time.perf_counter()
                save_training_figure(
                    self._debug_train_dir / f"ep{epoch:03d}_step{step_idx:04d}.png",
                    x,
                    x_sparse,
                    x_hat_clipped,
                    psnr_db=step_psnr,
                    title=f"train epoch={epoch} step={step_idx}",
                )
                save_times.append(time.perf_counter() - _ts)

            # ---- optimiser step -----------------------------------------
            is_last = step_idx + 1 == total_steps
            if (step_idx + 1) % accum_steps == 0 or is_last:
                self._optimizer_step_and_clamp()

        mean_psnr = float(torch.tensor(psnr_vals).mean()) if psnr_vals else 0.0
        mean_loss = float(sum(loss_vals) / len(loss_vals)) if loss_vals else 0.0
        if is_rank0 and fwd_times:
            epoch_total = time.perf_counter() - _epoch_t0
            n = len(fwd_times)
            mean_fwd = sum(fwd_times) / n
            mean_bwd = sum(bwd_times) / n
            mean_other = (epoch_total - sum(fwd_times) - sum(bwd_times)) / n
            self.epoch_train_total_times.append(epoch_total)
            self.epoch_fwd_times.append(mean_fwd)
            self.epoch_bwd_times.append(mean_bwd)
            self.epoch_other_times.append(mean_other)
            print(
                f"  [train epoch {epoch}] total={epoch_total:.1f}s ({n} steps) | "
                f"per step: fwd={mean_fwd:.3f}s bwd={mean_bwd:.3f}s other={mean_other:.3f}s",
                flush=True,
            )
        return mean_psnr, mean_loss

    def evaluate(
        self,
        dataloader,
        epoch: int,
        *,
        cache_as_init: bool = False,
        save_predictions: bool = False,
    ) -> "tuple[float, float]":
        """Evaluate the model on *dataloader* without gradient updates.

        On the first batch the tuple ``(x_cpu, xs_cpu, xh_cpu)`` is stored in
        :attr:`_cached_val_batch` so the solver can visualise it later.
        When *cache_as_init* is ``True`` the same tuple is also stored in
        :attr:`_cached_init_val_batch`, and each step's ``x_hat_clipped`` is
        cached in :attr:`_init_images` keyed by ``(step_idx, num_proj_key)``
        so subsequent val runs can include an "untrained" panel per sample.

        A debug figure (GT | Sparse | Untrained | Pred) is saved to
        ``debug_val/`` every ``max(1, save_debug_every)`` steps (rank-0 only).

        For tomo datasets (``num_proj_key is not None``) the first batch's
        reconstruction and ground truth are also saved as ``.pt`` volume files
        when *save_predictions* is ``True``.

        Returns
        -------
        (mean_psnr, mean_loss) over all batches.
        """
        self.model.eval()
        psnr_vals: list[float] = []
        loss_vals: list[float] = []
        is_rank0 = self.ctx is None or self.ctx.rank == 0
        _val_t0 = time.perf_counter()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x, x_sparse, y_tl, num_proj_key, sample_opnorm = self._unwrap_batch(
                    batch
                )
                physics = self._get_physics(num_proj_key)

                if self.normalize:
                    x, _norm_min, _norm_scale = normalize_to_unit(x)
                    if x_sparse is not None:
                        x_sparse = (x_sparse - _norm_min) / _norm_scale

                _step_scale = 1.0 / max(sample_opnorm, 1e-8)
                self._apply_stepsize_scale(_step_scale)
                _init = x_sparse if (self.use_x_init and x_sparse is not None) else None
                x_hat = self.model(y_tl, physics, init=_init)
                self._apply_stepsize_scale(_step_scale, restore=True)

                x_hat_clipped = x_hat.clamp(min=self.min_pixel)
                step_psnr = float(self._compute_psnr(x_hat_clipped, x, num_proj_key))
                step_loss = float(torch.nn.functional.mse_loss(x_hat, x))
                psnr_vals.append(step_psnr)
                loss_vals.append(step_loss)
                self.all_val_loss_steps.append(step_loss)

                # ---- cache init images per step for later comparison ----
                cache_key = (i, num_proj_key)
                if cache_as_init:
                    self._init_images[cache_key] = x_hat_clipped.detach().cpu()
                    self._init_psnrs[cache_key] = step_psnr

                # ---- cache first-batch reconstruction ------------------
                if i == 0:
                    self._cached_val_batch = (
                        x.cpu(),
                        (
                            x_sparse.cpu()
                            if (self.use_x_init and x_sparse is not None)
                            else None
                        ),
                        x_hat_clipped.cpu(),
                    )
                    # Tomo-only: persist prediction snapshots to disk (only when requested).
                    if save_predictions and is_rank0 and num_proj_key is not None:
                        save_prediction_results(
                            x_hat_clipped,
                            self._predictions_dir / f"ep{epoch:03d}_xhat.pt",
                        )
                        save_prediction_results(
                            x,
                            self._predictions_dir / f"ep{epoch:03d}_xref.pt",
                        )

                # ---- per-step CSV logging -------------------------------
                if is_rank0:
                    np_key = num_proj_key if num_proj_key is not None else "-"
                    print(
                        f"  [val step {i + 1}] "
                        f"np={np_key} | opnorm={sample_opnorm:.3f} | "
                        f"loss={step_loss:.6f} | psnr={step_psnr:.2f} dB",
                        flush=True,
                    )
                    append_metrics_row(
                        self._metrics_dir / "val_steps.csv",
                        {
                            "epoch": epoch,
                            "step": i,
                            "psnr": step_psnr,
                            "loss": step_loss,
                        },
                    )

                # ---- debug figure for every val step --------------------
                if is_rank0 and (i + 1) % max(1, self.save_debug_every) == 0:
                    x_init_cpu = self._init_images.get(cache_key)
                    psnr_init = self._init_psnrs.get(cache_key)
                    x_init_dev = (
                        x_init_cpu.to(self.device) if x_init_cpu is not None else None
                    )
                    save_training_figure(
                        self._debug_val_dir / f"ep{epoch:03d}_step{i:04d}.png",
                        x,
                        x_sparse,
                        x_hat_clipped,
                        psnr_db=step_psnr,
                        title=f"val epoch={epoch} step={i} np={num_proj_key}",
                        x_init=x_init_dev,
                        psnr_init_db=psnr_init,
                    )

        if cache_as_init:
            self._cached_init_val_batch = self._cached_val_batch

        if is_rank0:
            val_total = time.perf_counter() - _val_t0
            n_val = len(psnr_vals)
            per_sample = val_total / max(1, n_val)
            self.epoch_val_times.append(val_total)
            self.epoch_val_per_sample_times.append(per_sample)
            print(
                f"  [val epoch {epoch}] total={val_total:.1f}s | "
                f"per sample={per_sample:.1f}s ({n_val} samples)",
                flush=True,
            )

        mean_psnr = float(torch.tensor(psnr_vals).mean()) if psnr_vals else 0.0
        mean_loss = float(sum(loss_vals) / len(loss_vals)) if loss_vals else 0.0
        return mean_psnr, mean_loss

    def save_final_predictions(self, epoch: int) -> None:
        """Persist the cached val-batch predictions to disk (call once after training ends).

        Saves ``xhat`` and ``xref`` from :attr:`_cached_val_batch` as ``.pt``
        files.  Does nothing if no batch has been cached yet (e.g. training
        was skipped) or if the trainer has no distributed rank-0 context.
        """
        is_rank0 = self.ctx is None or self.ctx.rank == 0
        if not is_rank0 or self._cached_val_batch is None:
            return
        x_cpu, _, xh_cpu = self._cached_val_batch
        save_prediction_results(
            xh_cpu, self._predictions_dir / f"ep{epoch:03d}_xhat_final.pt"
        )
        save_prediction_results(
            x_cpu, self._predictions_dir / f"ep{epoch:03d}_xref_final.pt"
        )
