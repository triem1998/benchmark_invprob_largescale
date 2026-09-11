from benchopt import BaseSolver
from benchopt.stopping_criterion import NoCriterion
from deepinv.distributed import DistributedContext

from toolsbench.invprob.base import InvProb
from toolsbench.profiler import create_profiler
from toolsbench.solver.equivariant import EquivariantSolver
from toolsbench.utils.solver_utils import (
    build_solver_name,
    get_device_from_context,
    setup_distributed_env,
)


class Solver(BaseSolver):
    """Self-supervised cryo-ET training (one training step per iteration).

    ``preset`` selects the method: a plain denoiser (``tomo_ei``) or a PGD
    unfold (``unrolled``).
    """

    name = "Equivariant"

    sampling_strategy = "callback"
    # Disable convergence checking — run for exactly max_runs training steps.
    stopping_criterion = NoCriterion()

    parameters = {
        # --- Method ---
        # tomo_ei = denoise each half's FBP volume; unrolled = PGD unfold.
        "preset": ["tomo_ei"],
        # --- Physics ---
        # auto | astra | torch | torch_exact. auto = astra where its CUDA
        # kernels can run, else torch. torch reproduces astra's approximate
        # adjoint (so switching backends leaves the gradient unchanged);
        # torch_exact uses the true transpose instead.
        "tomography_backend": ["auto"],
        # None = one full operator per rank; "auto" = one per rank; int = that
        # many. Capped at the half-set's tilt count.
        "num_operators": [None],
        # --- Model architecture ---
        "f_maps": [64],
        "num_levels": [4],
        # >0 flips the UNet layer order to "crd", inserting Dropout.
        "unet_dropout": [0.0],
        "compile_model": [False],
        # --- Loss / optimizer ---
        "eq_weight": [0.0],
        # Rescales A(x_net) onto y's scale: none | znorm | leastsq_xnet |
        # leastsq_xnet_frozen.
        "obs_gain": ["none"],
        # Weight the residual by |k|. Without it blur is cheap.
        "obs_ramp": [True],
        # Noise on Eq's simulated measurement, as a multiple of the measured
        # half-set level. 0 = clean, 1 = matched.
        "eq_noise": [0.0],
        # z-norm both Eq operands, so Eq scores shape and Obs owns amplitude.
        "eq_scale_free": [False],
        "learning_rate": [1e-4],
        "grad_clip": [1.0],
        # "off" = fp32; "fp16" / "bf16" autocast every denoiser pass (the
        # physics, the losses and the FSC always see fp32). Same spelling as
        # fp16 adds a GradScaler and reports amp_scale / amp_skipped:
        # a skipped step costs the same time as a taken one.
        "mixed_precision": ["off"],
        # torch.backends.cudnn.benchmark. On ROCm this drives MIOpen's kernel
        # search; leaving it off is what made conv3d backward 10-54x slower than
        # forward on MI300A. Needs ROCm >= 7 on AMD to help. See the solver.
        "cudnn_benchmark": [True],
        # --- Unrolled only; ignored by tomo_ei ---
        # PGD steps: the denoiser runs this many times per reconstruction, with
        # a data-fidelity gradient between each pair. The dominant cost axis.
        "n_iter": [2],
        # Used directly — both physics paths end up unit spectral norm.
        "init_stepsize": [0.9],
        # Learn the stepsize jointly with the denoiser.
        "train_algo_params": [False],
        # LR for the stepsize param group; None shares `learning_rate`.
        "stepsize_learning_rate": [None],
        # --- Metric ---
        "fsc_threshold": [0.143],
        "pixel_size": [1.0],
        # --- Distributed processing ---
        "distribute_model": [False],
        "patch_size": [128],
        "overlap": [16],
        "max_batch_size": [1],
        "checkpoint_batches": ["auto"],
        # --- SLURM / torchrun ---
        "slurm_nodes": [1],
        "slurm_ntasks_per_node": [1],
        "slurm_gres": ["gpu:1"],
        "torchrun_nproc_per_node": [1],
        # --- Distributed context ---
        # Ranks cooperating on one volume. null = all of them (tiling only).
        # N = N ranks per volume and world/N replicas, so a step covers
        # dp_world_size volumes: divide wall time by it before comparing.
        "inner_world_size": [None],
        "deterministic": [False],
        # --- Logging / profiling ---
        "name_prefix": ["equivariant"],
        "profiler_mode": ["custom"],
        "profiler_warmup": [0],
        "profiler_active": [0],
        "profiler_trace_dir": [None],
        "profiler_per_step": [True],
        "profiler_repeat": [1],
        "profiler_save_file": [False],
    }

    def set_objective(
        self,
        measurements,
        physics,
        ground_truth_shape,
        num_operators,
        ground_truth=None,
        min_pixel=0.0,
        max_pixel=1.0,
        **kwargs,
    ):
        self.problem = InvProb(
            ground_truth=ground_truth,
            measurements=measurements,
            physics=physics,
            ground_truth_shape=ground_truth_shape,
            num_operators=num_operators,
            min_pixel=min_pixel,
            max_pixel=max_pixel,
        )
        self.ctx = None
        self._algo = None
        self.world_size = setup_distributed_env()
        self.distributed_mode = self.world_size > 1
        self.name = build_solver_name(
            self.name_prefix,
            self.slurm_nodes,
            self.slurm_ntasks_per_node,
            self.torchrun_nproc_per_node,
            self.distributed_mode,
        )

    def run(self, cb):
        if self.distributed_mode:
            # seed_offset=False: the denoiser is replicated across ranks and
            # split by input tiles, so every rank must initialise the *same*
            # weights. deepinv's default seeds each rank as seed + rank, which
            # gives each one a different model (measured: differing parameter
            # checksums and a 300x gradient-norm spread across two ranks).
            with DistributedContext(
                seed=42,
                cleanup=True,
                deterministic=self.deterministic,
                seed_offset=False,
                inner_world_size=self.inner_world_size,
            ) as ctx:
                self.ctx = ctx
                self._run_with_context(cb, ctx)
        else:
            self._run_with_context(cb, ctx=None)

    def _run_with_context(self, cb, ctx):
        device = get_device_from_context(ctx)
        profiler = create_profiler(
            self.profiler_mode,
            device,
            self.name,
            warmup=self.profiler_warmup,
            active=self.profiler_active,
            trace_dir=self.profiler_trace_dir,
            per_step=self.profiler_per_step,
            repeat=self.profiler_repeat,
            save_file=self.profiler_save_file,
        )
        with profiler:
            self._algo = EquivariantSolver(
                problem=self.problem,
                device=device,
                profiler=profiler,
                ctx=ctx,
                distributed_mode=self.distributed_mode,
                preset=self.preset,
                tomography_backend=self.tomography_backend,
                num_operators=self.num_operators,
                eq_weight=self.eq_weight,
                obs_gain=self.obs_gain,
                obs_ramp=self.obs_ramp,
                eq_noise=self.eq_noise,
                eq_scale_free=self.eq_scale_free,
                learning_rate=self.learning_rate,
                grad_clip=self.grad_clip,
                f_maps=self.f_maps,
                num_levels=self.num_levels,
                unet_dropout=self.unet_dropout,
                compile_model=self.compile_model,
                distribute_model=self.distribute_model,
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
                fsc_threshold=self.fsc_threshold,
                pixel_size=self.pixel_size,
                mixed_precision=self.mixed_precision,
                cudnn_benchmark=self.cudnn_benchmark,
                n_iter=self.n_iter,
                init_stepsize=self.init_stepsize,
                train_algo_params=self.train_algo_params,
                stepsize_learning_rate=self.stepsize_learning_rate,
            )
            self._algo.run(cb)
        profiler.finalize(ctx)

    def get_result(self):
        if self._algo is None:
            return {"reconstruction": None}
        result = dict(name=self.name)
        result.update(self._algo.get_result())
        return result

    def get_next(self, stop_val):
        return stop_val + 1
