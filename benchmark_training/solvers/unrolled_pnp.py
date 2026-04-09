"""Unrolled PnP solver using deepinv's PGD optimizer with PnP prior."""

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.profiler as prof
from benchopt import BaseSolver
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss.metric import PSNR
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim import PGD
from deepinv.utils.tensorlist import TensorList

from toolsbench.utils import (
    create_drunet_denoiser,
    save_training_curves,
    save_reconstruction_figure,
)


class Solver(BaseSolver):
    """Unrolled PnP solver using deepinv's PGD optimizer with PnP prior and distributed support."""

    name = "UnrolledPnP"

    requirements = [
        "pip::torch",
        "numpy",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    # Use callback sampling strategy for transparent epoch control
    sampling_strategy = "callback"

    # Solver parameters
    parameters = {
        "denoiser": ["drunet"],
        "n_iter": [4],
        "init_stepsize": [0.8],
        "denoiser_sigma": [0.05],
        "learning_rate": [1e-5],
        "model_learning_rate": [1e-5],
        "distribute_model": [True],
        "torch_compile": [False],
        "patch_size": [128],
        "overlap": [32],
        "max_batch_size": [1],
        # Activation checkpointing for patch-batches in DistributedProcessing.
        "checkpoint_batches": ["auto"],
        "lambda_relaxation": [False],
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        "name_prefix": ["unrolled_pnp"],
        "profile_dir": ["tb_profiles"],
        "operator_norm": [1.0],
        # Set to k * len(num_projs) (e.g. 3k) so each optimizer update sees all configs.
        "grad_accumulation_steps": [1],
    }

    def set_objective(
        self,
        train_dataloader,
        val_dataloader,
        physics,
        ground_truth_shape,
        num_operators,
        min_pixel=0.0,
        max_pixel=1.0,
        operator_norm=1.0,
    ):
        """Set the objective from the dataset.

        `physics` and `num_operators` may be dicts keyed by num_proj (multi-config)
        or plain values (single-config, backward compatible).
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # Normalise to dict form so _run_with_context and _get_physics
        # have a single uniform code path for both single- and multi-config datasets.
        if isinstance(physics, dict):
            self.physics = physics
            self.num_operators = num_operators  # dict {num_proj: int}
        else:
            self.physics = {None: physics}
            self.num_operators = {None: num_operators}
        self.ground_truth_shape = ground_truth_shape
        self.min_pixel = float(min_pixel)
        # Dataset-provided operator norm (overridden by solver parameter if set)
        self._dataset_operator_norm = float(operator_norm)

        self.world_size = 1
        self.ctx = None

        # Check if distributed environment is already set up
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            print(
                f"Distributed environment already initialized: world_size={self.world_size}"
            )
        else:
            try:
                import submitit

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                print(
                    f"Initialized distributed environment via submitit: world_size={self.world_size}"
                )
            except (ImportError, RuntimeError) as e:
                print(f"Running in non-distributed mode: {e}")

        self.distributed_mode = self.world_size > 1
        self.val_reconstructions = []

        # Generate name
        if hasattr(self, "slurm_ntasks_per_node") and self.slurm_ntasks_per_node > 1:
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + f"{self.slurm_nodes}n{self.slurm_ntasks_per_node}t"
            )
        elif (
            hasattr(self, "torchrun_nproc_per_node")
            and self.torchrun_nproc_per_node > 1
        ):
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + f"torchrun_{self.torchrun_nproc_per_node}proc"
            )
        else:
            self.name = (
                self.name_prefix
                + datetime.now().strftime("_%Y%m%d_%H%M%S_")
                + "_single"
            )

        if self.distributed_mode:
            self.name = self.name + f"_rank{int(os.environ.get('RANK', 0))}"

    def run(self, cb):
        """Run the unrolled PGD training with callback for epoch control.

        Args:
            cb: Callback function to call at each epoch. Returns False when to stop.
        """
        with DistributedContext(seed=0, cleanup=True) as ctx:
            self.ctx = ctx
            self._run_with_context(cb, ctx)

    def _setup_components(self, device, ctx=None):
        """Setup denoiser, prior, data fidelity, and PGD model.

        Both denoiser weights (at model_learning_rate) and algorithmic
        parameters (stepsize, g_param/sigma, at learning_rate) are trainable.

        Args:
            device: Torch device to use.
            ctx: Optional distributed context.

        Returns:
            tuple: (model: PGD(unfold=True), denoiser_params: list of raw denoiser parameters).
        """
        if self.denoiser == "drunet":
            denoiser = create_drunet_denoiser(
                ground_truth_shape=self.ground_truth_shape,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown denoiser: {self.denoiser}")

        # Collect raw denoiser parameters before any wrapping (distribute / compile)
        # so the optimizer can access them regardless of wrapping.
        denoiser_params = list(denoiser.parameters())

        # distribute data fidelity separately
        data_fidelity = distribute(L2(), ctx)

        # Build PnP prior with plain denoiser
        prior = PnP(denoiser=denoiser)

        # Effective step size: init_stepsize / operator_norm
        # Use solver parameter if explicitly set (!=1.0), otherwise fall back to dataset value.
        eff_opnorm = (
            float(self.operator_norm)
            if float(self.operator_norm) != 1.0
            else self._dataset_operator_norm
        )
        eff_stepsize = float(self.init_stepsize) / max(eff_opnorm, 1e-8)
        print(
            f"[unrolled_pnp] operator_norm={eff_opnorm:.4f}, init_stepsize={self.init_stepsize}, effective_stepsize={eff_stepsize:.6f}"
        )

        # Trainable: stepsize, g_param (algo params) + denoiser weights.
        # Optionally also train beta (proximal relaxation) when lambda_relaxation=True.
        trainable_params = ["stepsize", "g_param"]
        if self.lambda_relaxation:
            trainable_params.append("beta")
        model = PGD(
            stepsize=[eff_stepsize] * self.n_iter,
            sigma_denoiser=self.denoiser_sigma,
            beta=[1.0] * self.n_iter,
            trainable_params=trainable_params,
            data_fidelity=data_fidelity,
            max_iter=self.n_iter,
            prior=prior,
            unfold=True,
        )

        # Distribute the whole model as one unit.
        if self.distribute_model:
            model = distribute(
                model,
                ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )

        # Optionally compile the (distributed) model
        if self.torch_compile:
            model = torch.compile(model)

        return model, denoiser_params

    def _run_with_context(self, cb, ctx=None):
        """Run training with optional distributed context.

        Args:
            cb: Callback function to control epoch iterations
            ctx: Optional distributed context
        """
        # Determine device
        if ctx is not None:
            self.device = ctx.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rank = ctx.rank if ctx is not None else 0

        # Setup PyTorch profiler (rank 0 only, disabled if profile_dir is empty)
        self.profiler = None
        profile_dir_str = str(self.profile_dir).strip()
        if profile_dir_str:
            profile_path = Path(profile_dir_str) / self.name
            profile_path.mkdir(parents=True, exist_ok=True)
            activities = [prof.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(prof.ProfilerActivity.CUDA)
            if rank == 0:
                self.profiler = prof.profile(
                    activities=activities,
                    schedule=prof.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=prof.tensorboard_trace_handler(
                        str(profile_path / f"rank{rank}")
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                self.profiler.start()
                print(
                    f"[profiler] traces will be written to {profile_path / f'rank{rank}'}"
                )

        # Setup physics — build one distributed physics per config key.
        self.distributed_physics = {
            key: distribute(
                factory,
                ctx,
                num_operators=self.num_operators[key],
                type_object="linear_physics",
                reduction="mean",
            )
            for key, factory in self.physics.items()
        }

        # Setup model components
        self.model, denoiser_params = self._setup_components(self.device, ctx)
        print("Components set up.")

        # Optimizer: algo params at learning_rate, denoiser weights at model_learning_rate
        # Cast to float: benchopt may pass scientific notation values (e.g. 1e-5) as strings.
        self.algo_params = (
            list(self.model.params_algo["stepsize"])
            + list(self.model.params_algo["g_param"])
            + (list(self.model.params_algo["beta"]) if self.lambda_relaxation else [])
        )
        optimizer = torch.optim.Adam(
            [
                {"params": self.algo_params, "lr": float(self.learning_rate)},
                {"params": denoiser_params, "lr": float(self.model_learning_rate)},
            ]
        )

        self.optimizer = optimizer
        self.psnr_fn = PSNR(reduction="mean")

        # Track training and validation history
        self.train_psnr_history = []
        self.val_psnr_history = []
        self.loss_step_history = []  # loss per gradient step
        self.loss_epoch_history = []  # epoch-mean loss
        self.gpu_metrics_history = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Pre-evaluate untrained model so benchopt step 0 shows real PSNR (not 0).
        # Also cache the first val batch for the "unrolled (init)" comparison column.
        init_train_psnr = self._eval_loop(self.train_dataloader)
        init_val_psnr = self._eval_loop(self.val_dataloader, cache_first_batch=True)
        # Store init reconstruction permanently for per-epoch figure comparison.
        self._cached_init_val_batch = getattr(self, "_cached_val_batch", None)
        self.val_psnr_history.append(init_val_psnr)
        self.train_psnr_history.append(init_train_psnr)
        if ctx is None or ctx.rank == 0:
            print(
                f"[init] val PSNR: {init_val_psnr:.2f} dB | train PSNR: {init_train_psnr:.2f} dB (untrained model)"
            )

        print("Starting training with callback-controlled epochs...")
        print(f"Unrolled PGD model has {self.n_iter} iterations per forward pass")

        # Run training epochs with callback control
        self._run_training_epochs(cb, ctx)

        print(f"\nTraining completed after {len(self.train_psnr_history)} epochs")

        # Stop profiler and print summary
        if self.profiler is not None:
            self.profiler.stop()
            print(
                self.profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
            )

        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if ctx is not None:
            ctx.barrier()

        # Print final parameters
        if ctx is None or ctx.rank == 0:
            final_steps = [
                f"{s.item():.4f}" for s in self.model.params_algo["stepsize"]
            ]
            final_sigma = self.model.params_algo["g_param"][0].item()
            print(f"Final trainable step sizes: {final_steps}")
            print(f"Final denoiser sigma: {final_sigma:.5f}")
            if self.lambda_relaxation:
                final_betas = [
                    f"{b.item():.4f}" for b in self.model.params_algo["beta"]
                ]
                print(f"Final trainable betas: {final_betas}")
            print(f"Final train PSNR: {self.train_psnr_history[-1]:.2f} dB")
            print(f"Final validation PSNR: {self.val_psnr_history[-1]:.2f} dB")

    # ------------------------------------------------------------------
    # Manual training / evaluation loops (multi-physics aware)
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_psnr(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        crop: int = 100,
    ) -> torch.Tensor:
        """PSNR on the central crop of a 3-D volume, normalised by the crop's pixel range.

        Avoids cone-beam edge artefacts and focuses on material rather than air.
        Both tensors are expected to have shape (B, C, D, H, W).
        The range is max-min of the *target* crop (scalar over the whole crop).
        """
        # Crop last three spatial dims
        c = slice(crop, -crop)
        tgt = x[..., c, c, c]  # (B, C, D', H', W')
        pred = x_hat[..., c, c, c]
        range_val = tgt.amax() - tgt.amin()  # scalar
        mse = torch.mean((tgt - pred) ** 2)
        return 10.0 * torch.log10(range_val**2 / mse.clamp(min=1e-12))

    def _compute_psnr(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        num_proj_key,
    ) -> torch.Tensor:
        """Dispatch to crop-PSNR (Walnut CT) or standard PSNR (Urban100)."""
        if num_proj_key is not None:
            return self._crop_psnr(x_hat, x)
        return self.psnr_fn(x_hat, x)

    def _unpack_batch(self, batch):
        """Normalise batch to (x, x_sparse, y_tl, num_proj_key) and move to device.

        Handles two formats:
          - 2-tuple (x, y_tl)                      — Urban100 / HDF5Dataset
          - 4-tuple (x, x_sparse, y_tl, num_proj)  — Walnut CT

        Returns:
            x            : ground-truth tensor (on self.device)
            x_sparse     : warm-start tensor (on self.device), or None
            y_tl         : TensorList of measurements (on self.device)
            num_proj_key : num_proj int for physics selection, or None
        """

        def _to(t):
            if t is None:
                return None
            if isinstance(t, (list, TensorList)):
                return TensorList([ti.to(self.device) for ti in t])
            return t.to(self.device)

        if len(batch) == 2:
            x, y_tl = batch
            return _to(x), None, _to(y_tl), None
        x, x_sparse, y_tl, num_proj_key = batch
        return _to(x), _to(x_sparse), _to(y_tl), num_proj_key

    def _get_physics(self, num_proj_key):
        """Return the distributed physics for a given num_proj key (or None for single-config)."""
        if num_proj_key in self.distributed_physics:
            return self.distributed_physics[num_proj_key]
        # single-config fallback: the only key is None
        return self.distributed_physics[None]

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch over the shuffled dataloader.

        Handles both 2-tuple batches (Urban100) and 4-tuple batches (Walnut CT).
        When x_sparse is present it is passed as init to warm-start PGD.

        Returns:
            (mean_psnr, mean_loss) over the epoch.
        """
        self.model.train()
        psnr_vals = []
        loss_vals = []
        accum_steps = max(1, int(self.grad_accumulation_steps))
        self.optimizer.zero_grad()
        total_steps = len(self.train_dataloader)
        ctx = self.ctx
        is_rank0 = ctx is None or ctx.rank == 0
        for step_idx, batch in enumerate(self.train_dataloader):
            x, x_sparse, y_tl, num_proj_key = self._unpack_batch(batch)
            physics = self._get_physics(num_proj_key)

            x_hat = (
                self.model(y_tl, physics, init=x_sparse)
                if x_sparse is not None
                else self.model(y_tl, physics)
            )
            loss = torch.nn.functional.mse_loss(x_hat, x)
            # Scale loss so the effective gradient magnitude is independent of accum_steps
            (loss / accum_steps).backward()

            step_loss = loss.item()
            loss_vals.append(step_loss)
            self.loss_step_history.append(step_loss)
            with torch.no_grad():
                # Clip to min_pixel for PSNR only (negative attenuation is unphysical).
                # Not applied before loss.backward() to preserve gradients.
                x_hat_clipped = x_hat.clamp(min=self.min_pixel)
                step_psnr = self._compute_psnr(x_hat_clipped, x, num_proj_key)
                psnr_vals.append(step_psnr)

            if is_rank0:
                np_key = num_proj_key if num_proj_key is not None else "-"
                print(
                    f"  [train step {step_idx + 1}/{total_steps}] "
                    f"np={np_key} | loss={step_loss:.6f} | psnr={float(step_psnr):.2f} dB"
                )

            # Optimizer step every accum_steps batches (or at end of epoch)
            is_last = step_idx + 1 == len(self.train_dataloader)
            if (step_idx + 1) % accum_steps == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                    ],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        mean_psnr = float(torch.tensor(psnr_vals).mean()) if psnr_vals else 0.0
        mean_loss = float(sum(loss_vals) / len(loss_vals)) if loss_vals else 0.0
        return mean_psnr, mean_loss

    def _eval_loop(self, dataloader, cache_first_batch: bool = False) -> float:
        """Evaluate model on a dataloader without gradient updates.

        Handles both 2-tuple batches (Urban100) and 4-tuple batches (Walnut CT).
        When x_sparse is present it is passed as init to warm-start PGD.

        Args:
            dataloader: dataloader to evaluate on.
            cache_first_batch: if True, stores (x, x_sparse, x_hat) of the first
                batch (CPU tensors) in ``self._cached_val_batch`` for visualization.

        Returns mean PSNR over all batches.
        """
        self.model.eval()
        psnr_vals = []
        if cache_first_batch:
            self._cached_val_batch = None
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x, x_sparse, y_tl, num_proj_key = self._unpack_batch(batch)
                physics = self._get_physics(num_proj_key)
                x_hat = (
                    self.model(y_tl, physics, init=x_sparse)
                    if x_sparse is not None
                    else self.model(y_tl, physics)
                )
                # Clip to min_pixel (negative attenuation is unphysical); no upper clip.
                x_hat_clipped = x_hat.clamp(min=self.min_pixel)
                psnr_vals.append(self._compute_psnr(x_hat_clipped, x, num_proj_key))
                if cache_first_batch and i == 0:
                    self._cached_val_batch = (
                        x.cpu(),
                        x_sparse.cpu() if x_sparse is not None else None,
                        x_hat_clipped.cpu(),
                    )
        return float(torch.tensor(psnr_vals).mean()) if psnr_vals else 0.0

    def _run_training_epochs(self, cb, ctx=None):
        """Run training epochs with callback control using a manual per-batch loop.

        Each callback iteration runs one full pass over the shuffled training set
        (one epoch). The train dataloader re-shuffles on each iteration, ensuring
        IID (walnut_id, num_proj) sampling across epochs.

        Args:
            cb: Callback function to control iterations
            ctx: Optional distributed context
        """
        epoch = 0
        while True:
            keep_going = cb()

            # Synchronize decision in distributed mode
            if self.distributed_mode and ctx is not None:
                decision = torch.tensor([float(keep_going)], device=self.device)
                ctx.broadcast(decision, src=0)
                keep_going = bool(decision.item())

            if not keep_going:
                break

            # Run one epoch
            train_psnr, epoch_loss = self._train_epoch()
            val_psnr = self._eval_loop(self.val_dataloader, cache_first_batch=True)

            # Clamp trainable parameters to valid ranges.
            with torch.no_grad():
                for s in self.model.params_algo["stepsize"]:
                    s.clamp_(min=1e-8)
                if self.lambda_relaxation:
                    for b in self.model.params_algo["beta"]:
                        b.clamp_(0.0, 1.0)

            # Advance profiler schedule
            if self.profiler is not None:
                self.profiler.step()

            # Capture GPU memory stats for this epoch
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                total_mb = (
                    torch.cuda.get_device_properties(self.device).total_memory / 1024**2
                )
                allocated_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                max_allocated_mb = (
                    torch.cuda.max_memory_allocated(self.device) / 1024**2
                )
                reserved_mb = torch.cuda.memory_reserved(self.device) / 1024**2
                available_mb = total_mb - reserved_mb
                self.gpu_metrics_history.append(
                    {
                        "epoch": epoch,
                        "gpu_memory_allocated_mb": allocated_mb,
                        "gpu_memory_max_allocated_mb": max_allocated_mb,
                        "gpu_memory_reserved_mb": reserved_mb,
                        "gpu_available_memory_mb": available_mb,
                    }
                )
                torch.cuda.reset_peak_memory_stats(self.device)

            self.val_psnr_history.append(val_psnr)
            self.train_psnr_history.append(train_psnr)
            self.loss_epoch_history.append(epoch_loss)

            # Save training curves and reconstruction figure (rank 0 only).
            # Barrier before and after ensures all ranks stay in sync
            if ctx is not None and self.distributed_mode:
                ctx.barrier()
            if ctx is None or ctx.rank == 0:
                plot_dir = Path("images_results") / self.name
                save_training_curves(
                    self.loss_step_history,
                    self.loss_epoch_history,
                    self.train_psnr_history,
                    self.val_psnr_history,
                    save_dir=plot_dir,
                    grad_accumulation_steps=int(self.grad_accumulation_steps),
                )
                if getattr(self, "_cached_val_batch", None) is not None:
                    x_vis, xs_vis, xh_vis = self._cached_val_batch
                    # x_hat_init: unrolled output with the *initial* (untrained) weights
                    xh_init = (
                        self._cached_init_val_batch[2]
                        if getattr(self, "_cached_init_val_batch", None) is not None
                        else None
                    )
                    init_psnr = (
                        self.val_psnr_history[0] if self.val_psnr_history else None
                    )
                    save_reconstruction_figure(
                        x_vis,
                        xs_vis,
                        xh_vis,
                        psnr_val=val_psnr,
                        epoch=epoch + 1,
                        save_dir=plot_dir,
                        x_hat_init=xh_init,
                        psnr_init=init_psnr,
                    )
            if ctx is not None and self.distributed_mode:
                ctx.barrier()

            # Print progress
            if ctx is None or ctx.rank == 0:
                steps = [f"{s.item():.4f}" for s in self.model.params_algo["stepsize"]]
                sigma = self.model.params_algo["g_param"][0].item()
                beta_str = ""
                if self.lambda_relaxation:
                    betas = [f"{b.item():.4f}" for b in self.model.params_algo["beta"]]
                    beta_str = f" | beta: {betas}"
                print(
                    f"Epoch {epoch + 1} | train PSNR: {train_psnr:.2f} dB | "
                    f"val PSNR: {val_psnr:.2f} dB | -val PSNR: {-val_psnr:.2f} | "
                    f"steps: {steps} | sigma: {sigma:.5f}{beta_str}"
                )
                algo_vals = [f"{p.item():.6f}" for p in self.algo_params]
                print(f"  algo_params: {algo_vals}")

            epoch += 1

    def get_result(self):
        """Return the reconstruction result."""
        result = dict(
            val_psnr=self.val_psnr_history[-1] if self.val_psnr_history else 0.0,
            train_psnr=self.train_psnr_history[-1] if self.train_psnr_history else 0.0,
        )
        if self.gpu_metrics_history:
            last = self.gpu_metrics_history[-1]
            result["gpu_memory_allocated_mb"] = last["gpu_memory_allocated_mb"]
            result["gpu_memory_max_allocated_mb"] = last["gpu_memory_max_allocated_mb"]
            result["gpu_memory_reserved_mb"] = last["gpu_memory_reserved_mb"]
            result["gpu_available_memory_mb"] = last["gpu_available_memory_mb"]
        return result

    def get_next(self, stop_val):
        """Get next number of epochs."""
        return stop_val + 1
