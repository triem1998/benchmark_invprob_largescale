from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import deepinv as dinv
import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss import SupLoss
from deepinv.loss.metric import PSNR

from .dataloader import CryoDataConfig, build_cryo_dataloaders
from toolscryo.plot_metrics import plot_metrics
from toolscryo.utils import (
    append_metrics_row,
    dump_config_json,
    ensure_dir,
    seed_everything,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunConfig:
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_supervised"
    input_dir: str = "./dataset/empiar-11058"
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_train_vols: int | None = None  # None = use all remaining after val split
    max_val_vols: int = 10  # explicit number of validation volumes
    target_shape: tuple[int, int, int] | None = (
        None  # (D, H, W) resample; None = no resize
    )
    seed: int = 0

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    learning_rate: float = 1e-4
    grad_clip: float | None = 1.0
    ckp_interval: int = 10  # checkpoint every N epochs
    eval_interval: int = 1

    # ── Distributed / patching ──────────────────────────────────────────────
    distribute_model: bool = True
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 2
    checkpoint_batches: str | int | None = "auto"
    grad_accumulation_steps: int = (
        1  # accumulate gradients over N batches before stepping
    )


class CsvTrainer(dinv.Trainer):
    """dinv.Trainer subclass that additionally logs every epoch to CSV files
    and saves mid-slice PNG reconstructions at each eval step.

    Creates two files (rank-0 only):
      - ``metrics/train_epochs.csv`` — train loss + gradient norm per epoch
      - ``metrics/val_epochs.csv``   — validation PSNR (and other eval metrics) per epoch

    Set ``trainer._metrics_dir``, ``trainer._images_dir``, and ``trainer._ckpt_dir``
    (Paths) after construction.  Set ``trainer._grad_accum_steps`` (int, default 1)
    to enable gradient accumulation over N batches before each optimizer step.
    """

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        """Adds gradient accumulation: accumulates over ``_grad_accum_steps`` batches."""
        # ── Timing: reset epoch timer on first batch of each epoch ───────────
        if train:
            if epoch != getattr(self, "_current_train_epoch", None):
                self._current_train_epoch = epoch
                self._train_epoch_start = time.perf_counter()
                self._train_batch_count = 0
            self._train_batch_count = getattr(self, "_train_batch_count", 0) + 1

        accum = max(1, getattr(self, "_grad_accum_steps", 1))
        if accum == 1:
            return super().compute_loss(
                physics, x, y, train=train, epoch=epoch, step=step
            )

        # Track how many batches have been seen in the current window.
        self._accum_count = getattr(self, "_accum_count", 0)
        at_window_start = self._accum_count % accum == 0
        self._accum_count += 1
        at_window_end = self._accum_count % accum == 0

        # Parent handles zero_grad / optimizer.step when step=True.
        # We replicate only what changes: loss scaling and gating of those calls.
        logs = {}
        if train and step and at_window_start:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad() if train else torch.no_grad():
            x_net = self.model_inference(y=y, physics=physics, x=x, train=train)
            loss_total = 0
            if train or self.compute_eval_losses:
                for k, loss_fn in enumerate(self.losses):
                    loss = loss_fn(
                        x=x,
                        x_net=x_net,
                        y=y,
                        physics=physics,
                        model=self.model,
                        epoch=epoch,
                    )
                    loss_total += loss.mean()
                    meters = (
                        self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    )
                    meters.update(loss.detach().cpu().numpy())
                    if len(self.losses) > 1:
                        logs[loss_fn.__class__.__name__] = meters.avg
                meters = (
                    self.logs_total_loss_train if train else self.logs_total_loss_eval
                )
                meters.update(loss_total.item())
                logs["TotalLoss"] = meters.avg

        if train:
            (loss_total / accum).backward()
            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg
            if step and at_window_end:
                self.optimizer.step()

        return loss_total, x_net, logs

    def compute_metrics(self, x, x_net, y, physics, logs, train=True, epoch=None):  # type: ignore[override]
        x_net, logs = super().compute_metrics(
            x, x_net, y, physics, logs, train=train, epoch=epoch
        )
        if not train and x_net is not None:
            self._save_val_image(epoch, x, y, x_net)
        return x_net, logs

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        # Only rank-0 writes CSV
        if not self.verbose:
            return

        metrics_dir = getattr(self, "_metrics_dir", None)
        if metrics_dir is None:
            return

        row = {
            "epoch": step,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        fname = "train_epochs.csv" if train else "val_epochs.csv"
        append_metrics_row(Path(metrics_dir) / fname, row)

        if train:
            t_train = time.perf_counter() - getattr(
                self, "_train_epoch_start", time.perf_counter()
            )
            n_batches = max(1, getattr(self, "_train_batch_count", 1))
            print(
                f"[time] train epoch={step}  total={t_train:.1f}s  per_image={t_train / n_batches:.2f}s",
                flush=True,
            )
            self._val_start = time.perf_counter()

            alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
            total_gb = (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                / 1024**3
            )
            print(
                f"[gpu] step={step}  max_alloc={alloc_gb:.2f} GB / {total_gb:.1f} GB",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            t_val = time.perf_counter() - getattr(
                self, "_val_start", time.perf_counter()
            )
            print(f"[time] val   epoch={step}  total={t_val:.1f}s", flush=True)

        # ── Manual checkpoint every ckp_interval epochs (rank 0 only) ──────
        # log_metrics_mlops with train=False fires once per eval pass = once per epoch.
        ckpt_dir = getattr(self, "_ckpt_dir", None)
        if not train and ckpt_dir is not None:
            ckp_interval = getattr(self, "ckp_interval", 10)
            ep = step
            if ep % ckp_interval == 0:
                ckpt_path = Path(ckpt_dir) / f"ckp_{ep:04d}.pth"
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved {ckpt_path}", flush=True)

    def _save_val_image(self, epoch, x, y, x_net) -> None:
        """Save mid-slice PNG for one validation image (rank 0 only)."""
        if not self.verbose:
            return
        images_dir = getattr(self, "_images_dir", None)
        if images_dir is None:
            return

        if epoch != getattr(self, "_plot_epoch", None):
            self._plot_epoch = epoch
            self._plot_img_idx = 0
        img_idx = self._plot_img_idx
        self._plot_img_idx += 1

        if img_idx == 0 and epoch == 0:
            print(
                f"[data] x shape={tuple(x.shape)}  y shape={tuple(y.shape)}  x_net shape={tuple(x_net.shape)}",
                flush=True,
            )

        def _mid_slice(t: torch.Tensor) -> "np.ndarray":
            v = t[0, 0].detach().cpu().float()
            if v.ndim == 3:
                v = v[v.shape[0] // 2]
            return v.numpy()

        gt = _mid_slice(x)
        meas = _mid_slice(y)
        pred = _mid_slice(x_net)

        # PSNR on full 3-D volume (not just the displayed slice)
        _gt_t = x[0:1].detach().cpu().float()
        _pred_t = x_net[0:1].detach().cpu().float()
        _meas_t = y[0:1].detach().cpu().float()
        psnr_val = float(PSNR(max_pixel=None)(_pred_t, _gt_t))
        psnr_meas = float(PSNR(max_pixel=None)(_meas_t, _gt_t))

        vmin, vmax = float(gt.min()), float(gt.max())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(
            axes,
            [gt, meas, pred],
            [
                "GT (icecream)",
                f"Input corrected ({psnr_meas:.2f} dB)",
                f"Pred ({psnr_val:.2f} dB)",
            ],
        ):
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Eval epoch {epoch + 1}  —  vol {img_idx:02d}  —  mid-slice")
        fig.tight_layout()
        out = Path(images_dir) / f"epoch{epoch + 1:04d}" / f"vol{img_idx:02d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)

    def plot(self, epoch, physics, x, y, x_net, train=True):  # type: ignore[override]
        """Suppress the default plot — images are saved in compute_metrics."""
        return


def run_training(cfg: RunConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", asdict(cfg))

    data_cfg = CryoDataConfig(
        input_dir=cfg.input_dir,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        target_shape=cfg.target_shape,
        seed=int(cfg.seed),
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_cryo_dataloaders(data_cfg)

        # ── Model ────────────────────────────────────────────────────────────
        backbone = dinv.models.UNet(
            in_channels=1,
            out_channels=1,
            scales=4,
            residual=True,
            batch_norm="biasfree",
            dim=3,
        ).to(ctx.device)

        if cfg.distribute_model:
            backbone = distribute(
                backbone,
                ctx,
                patch_size=tuple(int(v) for v in cfg.patch_size),
                overlap=tuple(int(v) for v in cfg.overlap),
                tiling_dims=(-3, -2, -1),
                max_batch_size=cfg.max_batch_size,
                checkpoint_batches=cfg.checkpoint_batches,
            )

        model = dinv.models.ArtifactRemoval(backbone, mode="direct")

        physics = dinv.physics.Physics().to(ctx.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
        accum = max(1, int(cfg.grad_accumulation_steps))

        trainer = CsvTrainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=data_bundle.val_loader,
            epochs=int(cfg.num_epochs),
            losses=SupLoss(),
            metrics=PSNR(max_pixel=None),
            online_measurements=False,
            device=ctx.device,
            save_path=None,  # intentionally None: Trainer's makedirs runs on ALL ranks
            # and crashes ranks 1+ with exist_ok=False → hang.
            # We handle checkpointing manually in log_metrics_mlops.
            ckp_interval=int(cfg.ckp_interval),
            eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip,
            check_grad=cfg.grad_clip is not None,
            plot_images=True,
            verbose=rank == 0,
            show_progress_bar=rank == 0,
            log_train_batch=False,
            optimizer_step_multi_dataset=False,  # lets compute_loss own zero_grad+step (needed for accum)
        )
        trainer._metrics_dir = ensure_dir(output_dir / "metrics")
        trainer._images_dir = ensure_dir(output_dir / "images")
        trainer._ckpt_dir = (
            ensure_dir(output_dir / "checkpoints") if rank == 0 else None
        )
        trainer._grad_accum_steps = accum

        trainer.train()

        # ── Save final checkpoint (rank 0) ───────────────────────────────────
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save(
                {
                    "epoch": cfg.num_epochs,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved final {ckpt_path}", flush=True)

            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")
