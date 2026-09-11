"""Self-supervised cryo-ET training, one training step per benchopt iteration.

``preset`` selects the method:

``tomo_ei``
    a plain denoiser applied to each half-set's FBP volume.
``unrolled``
    a PGD unfold of ``n_iter`` steps, measurement-conditioned.

Everything else — physics, losses, metric, training loop — is shared. The
preset decides how the model is built, how it is called, where autocast
attaches, what the optimiser sees and what the FSC scores; all six follow from
the one choice.

No benchopt dependency: construct, ``run(cb)``, ``get_result()``.
"""

import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.optim import PGD
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP

from deepinv.distributed.framework import DistributedReplicatedParameters

from toolsbench.utils.cryo import (
    AMP_DTYPES,
    AmpDenoiser,
    EqLoss,
    GpuFSC,
    ObsLoss,
    Rotate3D,
    build_cryo_pair,
    build_unet3d,
    fsc_resolution,
    reduce_metrics_max,
    split_sinogram,
    wants_sharding,
)
from toolsbench.utils.solver_utils import (
    clamp_stepsize,
    distributed_callback_iter,
    sync_and_barrier,
)

PRESETS = ("tomo_ei", "unrolled")


class EquivariantSolver:
    """Half-set equivariant-imaging training, one step per benchopt iteration."""

    def __init__(
        self,
        problem,
        device,
        profiler,
        ctx,
        distributed_mode,
        *,
        preset="tomo_ei",
        tomography_backend="auto",
        num_operators=None,
        eq_weight=0.0,
        obs_gain="none",
        obs_ramp=False,
        eq_noise=0.0,
        eq_scale_free=False,
        learning_rate=1e-4,
        grad_clip=1.0,
        f_maps=64,
        num_levels=4,
        unet_dropout=0.0,
        compile_model=False,
        distribute_model=False,
        patch_size=128,
        overlap=16,
        max_batch_size=1,
        checkpoint_batches="auto",
        fsc_threshold=0.143,
        pixel_size=1.0,
        mixed_precision="off",
        cudnn_benchmark=True,
        # unrolled only; ignored by tomo_ei.
        n_iter=2,
        init_stepsize=0.9,
        train_algo_params=False,
        stepsize_learning_rate=None,
    ):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}.")
        self.preset = preset
        self.n_iter = n_iter
        self.init_stepsize = init_stepsize
        self.train_algo_params = train_algo_params
        self.stepsize_learning_rate = stepsize_learning_rate
        self.problem = problem
        self.device = device
        self.profiler = profiler
        self.ctx = ctx
        self.distributed_mode = distributed_mode
        self.tomography_backend = tomography_backend
        self.num_operators = num_operators
        self.eq_weight = eq_weight
        self.obs_gain = obs_gain
        self.obs_ramp = obs_ramp
        self.eq_noise = eq_noise
        self.eq_scale_free = eq_scale_free
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.f_maps = f_maps
        self.num_levels = num_levels
        self.unet_dropout = unet_dropout
        self.compile_model = compile_model
        self.distribute_model = distribute_model
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.checkpoint_batches = checkpoint_batches
        self.fsc_threshold = fsc_threshold
        self.pixel_size = pixel_size
        self.mixed_precision = mixed_precision
        # On ROCm this drives MIOpen's kernel search. Off by default in torch,
        # which is what made conv3d backward 10-54x slower than forward on
        # MI300A (184 ms -> 25 ms with it on). Needs ROCm >= 7.
        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.cudnn_benchmark = cudnn_benchmark
        if mixed_precision not in AMP_DTYPES:
            raise ValueError(
                f"mixed_precision must be one of {list(AMP_DTYPES)}, "
                f"got {mixed_precision!r}."
            )
        self._amp_dtype = AMP_DTYPES[mixed_precision]
        self._amp_device = torch.device(device).type
        # fp16 only: bf16 has fp32's exponent range, so scaling buys nothing
        # and its 65536x multiply is what overflowed at native resolution.
        self._scaler = (
            torch.amp.GradScaler(self._amp_device)
            if mixed_precision == "fp16"
            else None
        )
        self.amp_skipped = 0
        #: Profiler metrics for the last iteration, reduced across ranks.
        self._metrics = {}

        self.module = None
        self.reconstruction = None
        self.fsc_res = None
        self.fsc_shell = None
        self.loss = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_physics(self):
        """Both half-set operators, sharded when ``num_operators`` asks.

        Measurements are split to match the shards: the Obs loss compares
        ``A(x)`` against them shard for shard.
        """
        pair = build_cryo_pair(
            self.problem.physics,
            self.problem.measurements,
            self.device,
            ctx=self.ctx,
            num_operators=self.num_operators,
            backend=self.tomography_backend,
        )
        y_evn, y_odd = (m.to(self.device) for m in self.problem.measurements)
        if pair.num_operators is not None:
            y_evn = split_sinogram(y_evn, pair.num_operators)
            y_odd = split_sinogram(y_odd, pair.num_operators)
        return pair, y_evn, y_odd

    def _setup_model(self, pair):
        """The model to call, the module owning its parameters, and a label.

        For ``unrolled`` the PGD is both: returning the bare denoiser as the
        module would leave a trainable ``stepsize`` outside the optimiser and
        the gradient clip, and the run would still train the denoiser — a
        silent failure.
        """
        denoiser, info = build_unet3d(
            self.device,
            f_maps=self.f_maps,
            num_levels=self.num_levels,
            dropout=self.unet_dropout,
        )
        if self.preset == "unrolled":
            # PGD unfold. init_stepsize is used directly: both physics paths
            # end up unit spectral norm, so nothing rescales per volume.
            n_iter = int(self.n_iter)
            trainable = ["stepsize"] if self.train_algo_params else []
            # A sharded operator's A_adjoint spans ranks, so the data-fidelity
            # gradient needs the matching collective; a plain L2 would silently
            # use only the local shard.
            data_fidelity = (
                distribute(L2(), self.ctx) if pair.num_operators is not None else L2()
            )
            model = PGD(
                stepsize=[float(self.init_stepsize)] * n_iter,
                sigma_denoiser=0.0,
                beta=[1.0] * n_iter,
                trainable_params=trainable,
                data_fidelity=data_fidelity,
                max_iter=n_iter,
                prior=PnP(denoiser=denoiser),
                unfold=True,
            ).to(self.device)
            # Compile the plain denoiser before the tiling wrapper, so the
            # compiled region is the network rather than the tiling loop.
            if self.compile_model:
                model.prior[0].denoiser = torch.compile(model.prior[0].denoiser)
            if self.distribute_model and self.ctx is not None:
                # Denoiser-only: the PGD keeps its plain physics, so each rank
                # runs the projection locally while the denoiser is tiled.
                model.prior[0].denoiser = distribute(
                    model.prior[0].denoiser,
                    self.ctx,
                    type_object="denoiser",
                    tiling_dims=(-3, -2, -1),
                    patch_size=self.patch_size,
                    overlap=self.overlap,
                    max_batch_size=self.max_batch_size,
                    checkpoint_batches=self.checkpoint_batches,
                )
            # Trainable algo params need a cross-rank sync: the tiled denoiser
            # makes each rank's contribution differ.
            algo_params = [
                p
                for v in model.params_algo.values()
                if isinstance(v, torch.nn.ParameterList)
                for p in v
            ]
            if algo_params and self.ctx is not None:
                model._deepinv_dist_sync = DistributedReplicatedParameters(
                    self.ctx, algo_params, average=True
                )
            return model, model, f"unrolled(PGD n_iter={n_iter}, {trainable}) {info}"

        model = denoiser
        if self.compile_model:
            model = torch.compile(model)
        module = model
        if self.distribute_model and self.ctx is not None:
            # tiling_dims are the spatial axes of (B, C, D, H, W).
            model = distribute(
                model,
                self.ctx,
                type_object="denoiser",
                tiling_dims=(-3, -2, -1),
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )
        return model, module, info

    def _amp(self, model):
        """``model`` under autocast, returning fp32. Identity when off.

        Wrapping once covers every denoiser pass — the step, ``EqLoss`` and
        ``_recon`` — so no call site can be missed.

        For ``unrolled`` it attaches to the denoiser *inside* the PGD: wrapping
        the whole call would put the PGD's own ``A``/``A_adjoint`` under
        autocast, and astra has no dtype guard.
        """
        if self._amp_dtype is None:
            return model
        if self.preset == "unrolled":
            model.prior[0].denoiser = AmpDenoiser(
                model.prior[0].denoiser, self._amp_device, self._amp_dtype
            )
            return model
        return AmpDenoiser(model, self._amp_device, self._amp_dtype)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, cb):
        # Both tiling the denoiser and sharding the physics need a context:
        # the sharded operator asks it which shards this rank holds.
        if (
            self.distribute_model or wants_sharding(self.num_operators)
        ) and self.ctx is None:
            self.ctx = DistributedContext()

        pair, y_evn, y_odd = self._setup_physics()
        model, module, info = self._setup_model(pair)
        self.module = module
        # Every model call below — step, EqLoss, _recon — goes through this.
        model = self._amp(model)
        optimizer = self._build_optimizer(module)
        transform = Rotate3D(volume_shape=pair.volume_shape)
        self._obs = ObsLoss(gain=self.obs_gain, ramp=self.obs_ramp)
        self._eq = EqLoss(
            transform,
            weight=float(self.eq_weight),
            unrolled=self.preset == "unrolled",
            noise=float(self.eq_noise),
            scale_free=bool(self.eq_scale_free),
        )
        self._fsc = GpuFSC(device=self.device)
        print(
            f"Components set up: {info}, backend={pair.backend}, "
            f"num_operators={pair.num_operators}."
        )

        # Untrained reconstruction, so benchopt's evaluation at step 0 has a
        # valid result — the same contract the unrolled solver follows.
        module.eval()
        with torch.no_grad():
            x0, y0 = self._forward(model, pair, y_evn, y_odd)
            self._score(model, pair, x0, y0)

        print(f"Starting {self.preset} training (one step per iteration).")
        module.train()
        for _ in distributed_callback_iter(
            cb, self.distributed_mode, self.device, self.ctx
        ):
            self._step(model, optimizer, pair, y_evn, y_odd)
            self.profiler.end_iteration(self.ctx)
            # Inside the loop, which ``distributed_callback_iter`` gates with a
            # broadcast, so every rank reaches this exactly once per iteration.
            self._metrics = reduce_metrics_max(
                self.profiler.get_current_metrics(), self.ctx
            )
        sync_and_barrier(self.device, self.ctx)

    # ------------------------------------------------------------------
    # Preset-dependent behaviour
    # ------------------------------------------------------------------

    def _build_optimizer(self, module):
        """Adam, with a separate group for the stepsize when it is learned."""
        lr = float(self.learning_rate)
        if (
            self.preset != "unrolled"
            or not self.train_algo_params
            or self.stepsize_learning_rate is None
        ):
            return torch.optim.Adam(module.parameters(), lr=lr)
        stepsize = list(module.params_algo["stepsize"])
        stepsize_ids = {id(p) for p in stepsize}
        others = [p for p in module.parameters() if id(p) not in stepsize_ids]
        return torch.optim.Adam(
            [
                {"params": others, "lr": lr},
                {"params": stepsize, "lr": float(self.stepsize_learning_rate)},
            ]
        )

    def _forward(self, model, pair, y_evn, y_odd):
        """Each half's reconstruction.

        ``tomo_ei`` denoises the FBP volume; ``unrolled`` reconstructs from the
        measurements with that volume as the PGD's ``x_0``. ``y_evn``/``y_odd``
        are already split per shard by :meth:`_setup_physics`.
        """
        if self.preset == "unrolled":
            return (
                model(y_evn, pair.physics_evn, init=pair.init_evn),
                model(y_odd, pair.physics_odd, init=pair.init_odd),
            )
        return model(pair.init_evn), model(pair.init_odd)

    def _recon(self, model, pair, x_net, y_net):
        """What the FSC scores.

        ``tomo_ei`` uses the two-pass round trip ``f(fbp(A(f(.))))``
        which imprints the real missing-angle
        pattern. ``unrolled`` returns the pair unchanged: the PGD already ran
        the measurement-consistency step ``n_iter`` times, so its ``fsc``
        profiler region is not comparable with ``tomo_ei``'s.
        """
        if self.preset == "unrolled":
            return x_net, y_net
        return (
            model(pair.physics_evn.fbp(pair.physics_evn.A(x_net))),
            model(pair.physics_odd.fbp(pair.physics_odd.A(y_net))),
        )

    def _post_optimizer_step(self):
        """Keep a learned stepsize positive. No-op for a plain denoiser."""
        if self.preset == "unrolled":
            clamp_stepsize(self.module)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def _step(self, model, optimizer, pair, y_evn, y_odd):
        optimizer.zero_grad(set_to_none=True)

        with self.profiler.track_step("forward"):
            # model is the _amp wrapper: it autocasts the denoiser and hands
            # back fp32, so the physics and the losses always see fp32.
            x_net, y_net = self._forward(model, pair, y_evn, y_odd)
            loss = self._obs(pair, x_net, y_net, y_evn, y_odd)
            if self.eq_weight > 0:
                loss = loss + self._eq(pair, model, x_net, y_net, y_evn, y_odd)

        with self.profiler.track_step("backward"):
            (self._scaler.scale(loss) if self._scaler is not None else loss).backward()

        if self._scaler is not None:
            # Clip the true gradients, not the scaled ones.
            self._scaler.unscale_(optimizer)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), float(self.grad_clip)
            )
        if self._scaler is None:
            optimizer.step()
        else:
            prev_scale = self._scaler.get_scale()
            self._scaler.step(optimizer)
            self._scaler.update()
            # A dropped step (overflow) costs a full step's time, so count it.
            self.amp_skipped += prev_scale > self._scaler.get_scale()
        self._post_optimizer_step()
        self.loss = loss.item()

        # Timed like any other region: under tomo_ei the score costs another
        # A + fbp + denoiser per half.
        with self.profiler.track_step("fsc"):
            with torch.no_grad():
                self._score(model, pair, x_net.detach(), y_net.detach())

    def _score(self, model, pair, x_net, y_net):
        """FSC between the two half-set reconstructions.

        Scored apart, never pre-averaged: FSC measures agreement between two
        independent half-sets. What is scored comes from :meth:`_recon`.
        """
        r_evn, r_odd = self._recon(model, pair, x_net, y_net)
        curve = self._fsc(r_evn, r_odd)
        shell, resolution, _ = fsc_resolution(
            curve, r_evn.squeeze().shape, self.pixel_size, self.fsc_threshold
        )
        self.fsc_shell, self.fsc_res = int(shell), float(resolution)
        self.reconstruction = 0.5 * (r_evn + r_odd)

    def get_result(self):
        result = dict(
            reconstruction=self.reconstruction,
            fsc_res=self.fsc_res,
            fsc_shell=self.fsc_shell,
            train_loss=self.loss,
            # Both None off the fp16 path; the objective drops None values, so
            # these become columns only where they mean something.
            amp_scale=self._scaler.get_scale() if self._scaler is not None else None,
            amp_skipped=self.amp_skipped if self._scaler is not None else None,
        )
        if self.ctx is not None:
            # A step covers dp_world_size volumes: wall time is comparable
            # across topologies only once divided by it.
            result["inner_world_size"] = int(getattr(self.ctx, "inner_world_size", 1))
            result["dp_world_size"] = int(getattr(self.ctx, "dp_world_size", 1))
        if self.profiler is not None:
            # Reduced in the run loop, where the ranks are provably in step.
            result.update(self._metrics or self.profiler.get_current_metrics())
        return result
