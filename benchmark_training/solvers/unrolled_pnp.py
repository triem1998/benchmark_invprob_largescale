"""Unrolled PnP solver using deepinv's PGD optimizer with PnP prior."""

import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.profiler as prof
from benchopt import BaseSolver
from benchopt.stopping_criterion import NoCriterion
from deepinv.distributed import DistributedContext, distribute
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim import PGD

from toolsbench.utils import (
    _Trainer,
    TrainingHistory,
    create_drunet_denoiser,
    setup_distributed_env,
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

    # Disable convergence checking — run for exactly max_runs epochs
    stopping_criterion = NoCriterion()
    # Solver parameters
    parameters = {
        # --- Model architecture ---
        "denoiser": ["drunet"],  # Denoiser backbone ("drunet").
        "n_iter": [4],  # Number of unrolled PGD iterations.
        "init_stepsize": [0.8],  # Initial step size (scaled by 1/opnorm per batch).
        "denoiser_sigma": [0.05],  # Noise level fed to the denoiser.
        # --- Optimizer ---
        "learning_rate": [1e-5],  # LR for algorithmic parameters (stepsize, g_param).
        "model_learning_rate": [1e-5],  # LR for denoiser weights.
        # --- Distributed processing ---
        "distribute_model": [True],
        # Wrap model in DistributedProcessing (patch-based).
        "torch_compile": [False],  # Apply torch.compile to the model.
        "patch_size": [128],  # Spatial patch size for DistributedProcessing.
        "overlap": [32],  # Overlap between adjacent patches.
        "max_batch_size": [1],  # Max patch-batch size inside DistributedProcessing.
        "checkpoint_batches": ["auto"],  # Activation checkpointing for patch-batches
        # --- Algo options ---
        "lambda_relaxation": [False],
        # Also train beta (proximal relaxation parameter).
        # --- SLURM / torchrun ---
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        # --- Logging / profiling ---
        "name_prefix": ["unrolled_pnp"],
        # Prefix for the run name (timestamp appended).
        "profile_dir": ["tb_profiles"],  # Directory for TensorBoard profiler traces.
        "use_profiler": [False],  # Enable torch.profiler tracing (rank 0 only).
        # --- Physics ---
        "operator_norm": [1.0],  # Reference operator norm (overrides dataset value).
        # --- Training loop ---
        "grad_accumulation_steps": [1],
        # Accumulate gradients over N steps before optimizer.step().
        "train_algo_params": [True],
        # Also train stepsize/g_param; False = denoiser only.
        "normalize": [False],  # Normalize ground truth to [0, 1] before forward pass.
        "use_x_init": [True],  # Warm-start PGD from x_sparse when available.
        "max_batches_per_epoch": [None],  # Cap batches per epoch (None = full dataset).
        "clip_grad_norm": [1.0],  # Max gradient norm (0 = disabled).
        "eval_every": [1],  # Run validation every N epochs.
        "log_every": [1],  # Print/log metrics every N train steps.
        "save_debug_every": [0],  # Save debug figures every N steps (0 = disabled).
        "lr_decay_rate": [0.0],  # Exponential LR decay per epoch (0 = constant LR).
        "lr_min": [5e-6],        # LR floor: decay stops when LR reaches this value.
        # --- Reproducibility ---
        "seed": [0],  # RNG seed.
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
        operator_norm_map=None,
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
        self._ref_opnorm = (
            self._dataset_operator_norm
        )  # fallback for tuple-format batches
        self.operator_norm_map = (
            operator_norm_map if operator_norm_map is not None else {}
        )

        self.ctx = None
        self.world_size = setup_distributed_env()

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
        with DistributedContext(
            seed=self.seed, cleanup=True, deterministic=True
        ) as ctx:
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

        # Keep a direct reference to the unwrapped denoiser for checkpointing.
        self._denoiser = denoiser

        # distribute data fidelity separately
        data_fidelity = distribute(L2(), ctx)

        # Build PnP prior with plain denoiser
        prior = PnP(denoiser=denoiser)

        # Effective step size: init_stepsize / operator_norm
        eff_stepsize = float(self.init_stepsize)
        print(
            f"[unrolled_pnp] init_stepsize={self.init_stepsize} (stored); effective step = stepsize / sample_opnorm per batch"
        )

        # Trainable: stepsize, g_param (algo params) + denoiser weights.
        # Optionally also train beta (proximal relaxation) when lambda_relaxation=True.
        # When train_algo_params=False, only the denoiser is trained.
        if self.train_algo_params:
            trainable_params = ["stepsize", "g_param"]
            if self.lambda_relaxation:
                trainable_params.append("beta")
        else:
            trainable_params = []
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
        if self.use_profiler and profile_dir_str:
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
        _t = time.perf_counter()
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
        print(f"[timing] physics setup: {time.perf_counter()-_t:.1f}s", flush=True)

        # Setup model components
        _t = time.perf_counter()
        self.model, denoiser_params = self._setup_components(self.device, ctx)
        print(f"[timing] _setup_components: {time.perf_counter()-_t:.1f}s", flush=True)
        print("Components set up.")

        # Optimizer: algo params at learning_rate, denoiser weights at model_learning_rate
        # Cast to float: benchopt may pass scientific notation values (e.g. 1e-5) as strings.
        _t = time.perf_counter()
        if self.train_algo_params:
            self.algo_params = (
                list(self.model.params_algo["stepsize"])
                + list(self.model.params_algo["g_param"])
                + (
                    list(self.model.params_algo["beta"])
                    if self.lambda_relaxation
                    else []
                )
            )
            optimizer = torch.optim.Adam(
                [
                    {"params": self.algo_params, "lr": float(self.learning_rate)},
                    {"params": denoiser_params, "lr": float(self.model_learning_rate)},
                ]
            )
        else:
            self.algo_params = []
            optimizer = torch.optim.Adam(
                [{"params": denoiser_params, "lr": float(self.model_learning_rate)}]
            )
        print(f"[timing] optimizer setup: {time.perf_counter()-_t:.1f}s", flush=True)

        self.optimizer = optimizer
        if float(self.lr_decay_rate) > 0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(self.lr_decay_rate))
            print(f"[lr_schedule] exponential decay: gamma={self.lr_decay_rate} lr_min={self.lr_min}", flush=True)
        else:
            self.scheduler = None
            print("[lr_schedule] constant LR", flush=True)

        # Track training and validation history
        self.history = TrainingHistory()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Instantiate Trainer — handles per-batch/epoch loop, CSV logging,
        # debug images and volume saving.
        self._trainer = _Trainer(
            model=self.model,
            optimizer=self.optimizer,
            physics_map=self.distributed_physics,
            num_operators_map=self.num_operators,
            device=self.device,
            output_dir=Path("images_results") / self.name,
            min_pixel=self.min_pixel,
            ctx=ctx,
            grad_accumulation_steps=self.grad_accumulation_steps,
            clip_grad_norm=self.clip_grad_norm,
            max_batches_per_epoch=self.max_batches_per_epoch,
            log_every=self.log_every,
            save_debug_every=self.save_debug_every,
            use_x_init=self.use_x_init,
            normalize=self.normalize,
            train_algo_params=self.train_algo_params,
            lambda_relaxation=self.lambda_relaxation,
            ref_opnorm=self._dataset_operator_norm,
        )
        # Shared references so _run_training_epochs can access loss histories.
        self.history.loss_steps = self._trainer.all_loss_steps
        self.history.val_loss_steps = self._trainer.all_val_loss_steps

        # Pre-evaluate untrained model on val set so benchopt step 0 shows real PSNR.
        # Also caches the first val batch for the "unrolled (init)" comparison figure.
        _t = time.perf_counter()
        init_val_psnr, _ = self._trainer.evaluate(
            self.val_dataloader, epoch=-1, cache_as_init=True
        )
        print(f"[timing] pre-eval val: {time.perf_counter()-_t:.1f}s", flush=True)
        # Store init reconstruction permanently for per-epoch figure comparison.
        self._trainer._cached_init_val_batch = self._trainer._cached_val_batch
        self.history.val_psnr.append(init_val_psnr)
        if ctx is None or ctx.rank == 0:
            print(f"[init] val PSNR: {init_val_psnr:.2f} dB (untrained model)")

        print("Starting training with callback-controlled epochs...")
        print(f"Unrolled PGD model has {self.n_iter} iterations per forward pass")

        # Run training epochs with callback control
        _t = time.perf_counter()
        self._run_training_epochs(cb, ctx)
        print(f"[timing] total training: {time.perf_counter()-_t:.1f}s", flush=True)

        print(f"\nTraining completed after {len(self.history.train_psnr)} epochs")

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
            if self.train_algo_params:
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
            print(f"Final train PSNR: {self.history.train_psnr[-1]:.2f} dB")
            print(f"Final validation PSNR: {self.history.val_psnr[-1]:.2f} dB")

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
        best_val_psnr = float("-inf")
        self._last_psnr_by_key: dict = {}
        is_rank0 = ctx is None or ctx.rank == 0
        best_model_dir = Path("images_results") / self.name
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = best_model_dir / "best_drunet.pth"
        while True:
            keep_going = cb()

            # Synchronize decision in distributed mode
            if self.distributed_mode and ctx is not None:
                decision = torch.tensor([float(keep_going)], device=self.device)
                ctx.broadcast(decision, src=0)
                keep_going = bool(decision.item())

            if not keep_going:
                break

            # Advance the sampler epoch so each training epoch gets a fresh shuffle.
            sampler = getattr(self.train_dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            # Run one epoch
            train_psnr, epoch_loss = self._trainer.train_epoch(
                self.train_dataloader, epoch
            )

            # Step LR scheduler.
            if self.scheduler is not None:
                self.scheduler.step()
                # Clamp to lr_min floor.
                for pg in self.optimizer.param_groups:
                    pg["lr"] = max(pg["lr"], float(self.lr_min))
                if is_rank0:
                    current_lr = self.optimizer.param_groups[-1]["lr"]
                    print(f"[lr_schedule] epoch {epoch + 1}: lr={current_lr:.2e}", flush=True)

            if (epoch + 1) % max(1, int(self.eval_every)) == 0:
                val_psnr, val_loss = self._trainer.evaluate(self.val_dataloader, epoch)
                # Save best DRUNet denoiser checkpoint (rank 0 only)
                if (ctx is None or ctx.rank == 0) and val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    torch.save(self._denoiser.state_dict(), best_model_path)
                    print(
                        f"[checkpoint] New best val PSNR: {best_val_psnr:.2f} dB — "
                        f"DRUNet saved to {best_model_path}"
                    )
            else:
                val_psnr = float("nan")
                val_loss = float("nan")

            # Advance profiler schedule
            if self.profiler is not None:
                self.profiler.step()

            # Store per-key PSNR snapshot (populated by _Trainer.evaluate)
            self._last_psnr_by_key = dict(self._trainer.last_psnr_by_key)

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
                self.history.gpu_metrics.append(
                    {
                        "epoch": epoch,
                        "gpu_memory_allocated_mb": allocated_mb,
                        "gpu_memory_max_allocated_mb": max_allocated_mb,
                        "gpu_memory_reserved_mb": reserved_mb,
                        "gpu_available_memory_mb": available_mb,
                    }
                )
                torch.cuda.reset_peak_memory_stats(self.device)

            self.history.val_psnr.append(val_psnr)
            self.history.train_psnr.append(train_psnr)
            self.history.loss_epochs.append(epoch_loss)
            self.history.val_loss_epochs.append(val_loss)

            # Barrier before printing progress
            if ctx is not None and self.distributed_mode:
                ctx.barrier()

            # Print progress
            if ctx is None or ctx.rank == 0:
                if self.train_algo_params:
                    steps = [
                        f"{s.item():.4f}" for s in self.model.params_algo["stepsize"]
                    ]
                    sigma = self.model.params_algo["g_param"][0].item()
                    beta_str = ""
                    if self.lambda_relaxation:
                        betas = [
                            f"{b.item():.4f}" for b in self.model.params_algo["beta"]
                        ]
                        beta_str = f" | beta: {betas}"
                    print(
                        f"Epoch {epoch + 1} | train PSNR: {train_psnr:.2f} dB | "
                        f"val PSNR: {val_psnr:.2f} dB | -val PSNR: {-val_psnr:.2f} | "
                        f"steps: {steps} | sigma: {sigma:.5f}{beta_str}"
                    )
                    algo_vals = [f"{p.item():.6f}" for p in self.algo_params]
                    print(f"  algo_params: {algo_vals}")
                else:
                    print(
                        f"Epoch {epoch + 1} | train PSNR: {train_psnr:.2f} dB | "
                        f"val PSNR: {val_psnr:.2f} dB | -val PSNR: {-val_psnr:.2f}"
                    )
                # Print per-key PSNR breakdown when multiple keys are present
                if len(self._last_psnr_by_key) > 1:
                    parts = "  |  ".join(
                        f"np={k}: {v:.2f} dB"
                        for k, v in sorted(
                            self._last_psnr_by_key.items(),
                            key=lambda kv: (kv[0] is None, kv[0]),
                        )
                    )
                    print(f"  per-key val PSNR: {parts}")

            epoch += 1

        # Save prediction snapshots once, for the last completed epoch.
        self._trainer.save_final_predictions(epoch - 1)

    def get_result(self):
        """Return metrics from the last completed epoch for benchopt.

        Returns a dict with: ``val_psnr``, ``train_psnr``, ``train_loss``,
        ``val_loss``, ``mean_load_time``, ``mean_fwd_time``, ``mean_bwd_time``
        (per-step averages over the last epoch), and GPU memory stats when
        CUDA is available.
        """
        result = dict(
            val_psnr=self.history.val_psnr[-1] if self.history.val_psnr else 0.0,
            train_psnr=self.history.train_psnr[-1] if self.history.train_psnr else 0.0,
            train_loss=(
                self.history.loss_epochs[-1]
                if self.history.loss_epochs
                else float("nan")
            ),
            val_loss=(
                next(
                    (v for v in reversed(self.history.val_loss_epochs) if not (v != v)),
                    float("nan"),
                )
                if self.history.val_loss_epochs
                else float("nan")
            ),
            train_total_time=(
                self._trainer.epoch_train_total_times[-1]
                if self._trainer.epoch_train_total_times
                else float("nan")
            ),
            mean_fwd_time=(
                self._trainer.epoch_fwd_times[-1]
                if self._trainer.epoch_fwd_times
                else float("nan")
            ),
            mean_bwd_time=(
                self._trainer.epoch_bwd_times[-1]
                if self._trainer.epoch_bwd_times
                else float("nan")
            ),
            mean_other_time=(
                self._trainer.epoch_other_times[-1]
                if self._trainer.epoch_other_times
                else float("nan")
            ),
            val_total_time=(
                self._trainer.epoch_val_times[-1]
                if self._trainer.epoch_val_times
                else float("nan")
            ),
            val_per_sample_time=(
                self._trainer.epoch_val_per_sample_times[-1]
                if self._trainer.epoch_val_per_sample_times
                else float("nan")
            ),
        )
        if self.history.gpu_metrics:
            last = self.history.gpu_metrics[-1]
            result["gpu_memory_allocated_mb"] = last["gpu_memory_allocated_mb"]
            result["gpu_memory_max_allocated_mb"] = last["gpu_memory_max_allocated_mb"]
            result["gpu_memory_reserved_mb"] = last["gpu_memory_reserved_mb"]
            result["gpu_available_memory_mb"] = last["gpu_available_memory_mb"]
        # Per-key PSNR (e.g. np=30/50/100 for tomography-3D)
        psnr_by_key = getattr(self, "_last_psnr_by_key", {})
        for k, v in psnr_by_key.items():
            key_str = f"val_psnr_np{k}" if k is not None else "val_psnr_key_none"
            result[key_str] = v
        return result

    def get_next(self, stop_val):
        """Get next number of epochs."""
        return stop_val + 1
