"""Unrolled PnP solver using deepinv's PGD optimizer with PnP prior."""

import os
from datetime import datetime
from pathlib import Path

import torch
import torch.profiler as prof
import deepinv as dinv
from benchopt import BaseSolver
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss.metric import PSNR
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim import PGD

from toolsbench.utils import create_drunet_denoiser


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
        "distribute_physics": [False],
        "distribute_denoiser": [False],
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
        """Set the objective from the dataset."""
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.physics = physics
        self.ground_truth_shape = ground_truth_shape
        self.num_operators = num_operators
        self.clip_range = (min_pixel, max_pixel)
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

        # Distribute denoiser if requested
        if self.distribute_denoiser:
            denoiser = distribute(
                denoiser,
                ctx,
                patch_size=self.patch_size,
                overlap=self.overlap,
                tiling_dims=(
                    (-3, -2, -1) if len(self.ground_truth_shape) == 5 else (-2, -1)
                ),
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )

        # Optionally compile denoiser
        if self.torch_compile:
            denoiser = torch.compile(denoiser)

        # Create distributed data fidelity
        data_fidelity = L2()
        data_fidelity = distribute(data_fidelity, ctx)

        if self.torch_compile:
            data_fidelity = torch.compile(data_fidelity)

        # Build PnP prior with (distributed) denoiser
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

        # Setup physics - always use distribute() for consistency
        physics = distribute(
            self.physics,
            ctx,
            num_operators=self.num_operators,
            type_object="linear_physics",
            reduction="mean",
        )

        # Setup model components
        self.model, denoiser_params = self._setup_components(self.device, ctx)
        print("Components set up.")

        # Distribute trainable algorithmic parameters (stepsize and g_param)
        for i in range(len(self.model.params_algo["stepsize"])):
            self.model.params_algo["stepsize"][i] = distribute(
                self.model.params_algo["stepsize"][i], ctx
            )
        for i in range(len(self.model.params_algo["g_param"])):
            self.model.params_algo["g_param"][i] = distribute(
                self.model.params_algo["g_param"][i], ctx
            )
        if self.lambda_relaxation:
            for i in range(len(self.model.params_algo["beta"])):
                self.model.params_algo["beta"][i] = distribute(
                    self.model.params_algo["beta"][i], ctx
                )

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

        # Build Trainer — same components as the demo
        psnr_metric = PSNR(reduction="mean")
        self.trainer = dinv.Trainer(
            model=self.model,
            physics=physics,
            epochs=1,  # set to 1; train() is called once per callback iteration
            device=self.device,
            losses=[dinv.loss.SupLoss(metric=dinv.metric.MSE())],
            metrics=psnr_metric,
            optimizer=optimizer,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.val_dataloader,
            grad_clip=1.0,
            compute_train_metrics=True,
            compare_no_learning=False,
            save_path=None,
            verbose=(ctx is None or ctx.rank == 0),
            show_progress_bar=(ctx is None or ctx.rank == 0),
            check_grad=True,
        )

        # Track training and validation history
        self.train_psnr_history = []
        self.val_psnr_history = []
        self.gpu_metrics_history = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Pre-evaluate untrained model so benchopt step 0 shows real PSNR (not 0).
        init_train_psnr = self.trainer.test(
            self.train_dataloader, compare_no_learning=False
        ).get("PSNR", 0.0)
        init_val_psnr = self.trainer.test(
            self.val_dataloader, compare_no_learning=False
        ).get("PSNR", 0.0)
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

    def _run_training_epochs(self, cb, ctx=None):
        """Run training epochs with callback control using dinv.Trainer.

        Each callback iteration runs one epoch via trainer.train() with epochs=1.
        setup_train() always resets epoch_start=0, so epochs=1 means exactly one
        epoch per call. Model weights persist across calls.

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

            # Run one epoch via Trainer
            self.trainer.epochs = 1  # setup_train resets epoch_start=0 each call
            self.trainer.train()

            # Clamp beta to [0, 1] to keep the proximal relaxation meaningful.
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

            # Extract PSNR from Trainer history
            psnr_key = "PSNR"
            val_psnr = self.trainer.eval_metrics_history.get(psnr_key, [0.0])[-1]
            train_psnr = (
                self.trainer.train_metrics_history.get(psnr_key, [0.0])[-1]
                if self.trainer.train_metrics_history.get(psnr_key)
                else 0.0
            )
            self.val_psnr_history.append(val_psnr)
            self.train_psnr_history.append(train_psnr)

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
