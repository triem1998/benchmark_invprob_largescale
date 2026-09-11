import pytest
import torch
from unittest.mock import MagicMock, patch

from toolsbench.invprob import CryoEIInvProb
from toolsbench.invprob.base import InvProb, InvProbConfig
from toolsbench.profiler import NullProfiler
from toolsbench.solver.denoiser import DenoiserSolver
from toolsbench.solver.pnp import PnPSolver
from toolsbench.solver.equivariant import EquivariantSolver
from toolsbench.solver.unrolled_pnp import UnrolledPnPSolver
from toolsbench.utils.cryo import ObsLoss, build_cryo_pair
from toolsbench.utils.solver_utils import clamp_stepsize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_objective(**kwargs):
    defaults = dict(
        ground_truth=torch.zeros(1, 1, 8, 8),
        measurements=torch.zeros(1, 1, 8, 8),
        physics=MagicMock(),
        ground_truth_shape=torch.Size([1, 1, 8, 8]),
        num_operators=1,
    )
    defaults.update(kwargs)
    return InvProb(**defaults)


def _make_unrolled_solver(**kwargs):
    defaults = dict(
        problem=_make_objective(),
        device=torch.device("cpu"),
        profiler=NullProfiler(),
        ctx=None,
        distributed_mode=False,
        denoiser="drunet",
        n_iter=2,
        init_stepsize=0.8,
        denoiser_sigma=0.05,
        learning_rate=1e-5,
        model_learning_rate=1e-5,
        train_algo_params=True,
        lambda_relaxation=False,
        grad_clip=1.0,
        distribute_model=False,
        patch_size=128,
        overlap=32,
        max_batch_size=1,
        checkpoint_batches="auto",
        image_size=None,
    )
    defaults.update(kwargs)
    return UnrolledPnPSolver(**defaults)


def _make_equivariant_solver(**kwargs):
    """An equivariant solver built but never run — enough to check its setup."""
    defaults = dict(
        problem=None,
        device=torch.device("cpu"),
        profiler=NullProfiler(),
        ctx=None,
        distributed_mode=False,
        tomography_backend="torch",
    )
    defaults.update(kwargs)
    return EquivariantSolver(**defaults)


def _mock_denoiser():
    m = MagicMock()
    m.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    return m


def _make_solver(**attrs):
    """Return a PnPSolver with attributes set directly, bypassing __init__."""
    solver = PnPSolver.__new__(PnPSolver)
    defaults = dict(
        problem=_make_objective(),
        device=torch.device("cpu"),
        profiler=NullProfiler(),
        distributed_mode=False,
        ctx=None,
        norm_strategy="clip",
        denoiser_lambda_relaxation=None,
        denoiser_sigma=0.05,
        step_size=None,
        step_size_scale=0.99,
        denoiser="drunet",
        patch_size=128,
        overlap=32,
        max_batch_size=0,
        distribute_denoiser=False,
        distribute_physics=False,
        init_method="pseudo_inverse",
        compile=None,
        reconstruction=torch.zeros(1, 1, 8, 8),
        shape=(1, 1, 8, 8),
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(solver, k, v)
    return solver


def _run_one_iter(solver, prior, data_fidelity, step_size=0.1):
    """Run one PnP iteration using a patched distributed_callback_iter."""
    physics = MagicMock()
    measurements = torch.zeros_like(solver.reconstruction)
    with patch(
        "toolsbench.solver.pnp.distributed_callback_iter",
        return_value=iter([None]),
    ):
        solver._run_iterations(
            prior, data_fidelity, physics, measurements, step_size, None
        )


# ---------------------------------------------------------------------------
# PnPSolver._compute_step_size
# ---------------------------------------------------------------------------


class TestPnPSolverComputeStepSize:

    def test_float_step_size_returned_directly(self):
        solver = _make_solver(step_size=0.5, step_size_scale=0.99)
        assert solver._compute_step_size(MagicMock()) == 0.5

    def test_auto_step_size_calls_helper(self):
        solver = _make_solver(step_size=None, step_size_scale=0.5)
        with patch(
            "toolsbench.solver.pnp.compute_step_size_from_operator", return_value=2.0
        ) as mock_fn:
            result = solver._compute_step_size(MagicMock())
        mock_fn.assert_called_once()
        assert result == pytest.approx(1.0)  # 2.0 * 0.5


# ---------------------------------------------------------------------------
# PnPSolver._setup_components
# ---------------------------------------------------------------------------


class TestPnPSolverSetupComponents:

    def test_returns_prior_and_data_fidelity(self):
        from deepinv.optim.prior import PnP
        from deepinv.optim.data_fidelity import L2

        solver = _make_solver()
        with patch(
            "toolsbench.solver.pnp.create_drunet_denoiser", return_value=MagicMock()
        ):
            prior, data_fidelity = solver._setup_components()

        assert isinstance(prior, PnP)
        assert isinstance(data_fidelity, L2)

    def test_unknown_denoiser_raises(self):
        solver = _make_solver(denoiser="unknown_model")
        with pytest.raises(ValueError, match="Unknown denoiser"):
            solver._setup_components()


# ---------------------------------------------------------------------------
# PnPSolver._run_pnp_iterations
# ---------------------------------------------------------------------------


class TestPnPSolverIterations:

    def test_clip_no_relaxation_clamps_output(self):
        solver = _make_solver(
            norm_strategy="clip",
            denoiser_lambda_relaxation=None,
            reconstruction=torch.full((1, 1, 8, 8), 2.0),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.full((1, 1, 8, 8), 2.0)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        _run_one_iter(solver, prior, data_fidelity)

        assert solver.reconstruction.max().item() <= 1.0
        assert solver.reconstruction.min().item() >= 0.0

    def test_dynamic_no_relaxation_rescales(self):
        solver = _make_solver(
            norm_strategy="dynamic",
            denoiser_lambda_relaxation=None,
            reconstruction=torch.full((1, 1, 8, 8), 0.5),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.full((1, 1, 8, 8), 0.5)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        _run_one_iter(solver, prior, data_fidelity)

        assert solver.reconstruction.shape == torch.Size([1, 1, 8, 8])
        assert solver.reconstruction.mean().item() == pytest.approx(0.5, abs=1e-4)

    def test_dynamic_with_relaxation_alpha_blends(self):
        # reconstruction=0, grad=0, prox returns 1 → alpha-blend toward 1
        solver = _make_solver(
            norm_strategy="dynamic",
            denoiser_lambda_relaxation=1.0,
            denoiser_sigma=0.05,
            reconstruction=torch.zeros(1, 1, 8, 8),
            problem=_make_objective(min_pixel=0.0, max_pixel=1.0),
        )
        prior = MagicMock()
        prior.prox.return_value = torch.ones(1, 1, 8, 8)
        data_fidelity = MagicMock()
        data_fidelity.grad.return_value = torch.zeros(1, 1, 8, 8)

        step_size = 0.1
        _run_one_iter(solver, prior, data_fidelity, step_size=step_size)

        expected_alpha = (step_size * 1.0) / (1 + step_size * 1.0)
        assert solver.reconstruction.mean().item() == pytest.approx(
            expected_alpha, abs=1e-4
        )


# ---------------------------------------------------------------------------
# PnPSolver.get_result
# ---------------------------------------------------------------------------


class TestPnPSolverGetResult:

    def test_includes_profiler_metrics(self):
        solver = _make_solver(reconstruction=torch.ones(1, 1, 4, 4))
        solver.profiler = MagicMock()
        solver.profiler.get_current_metrics.return_value = {
            "total_time_sec": 0.5,
            "max_gpu_mb": 100.0,
        }
        result = solver.get_result()
        assert result["total_time_sec"] == 0.5
        assert result["max_gpu_mb"] == 100.0


# ---------------------------------------------------------------------------
# PnPSolver / DenoiserSolver image_size wiring
# ---------------------------------------------------------------------------


class TestSolverImageSizeResize:

    def test_pnp_resizes_with_device(self):
        sentinel = _make_objective()
        problem = MagicMock()
        problem.resized.return_value = sentinel
        device = torch.device("cpu")
        solver = PnPSolver(
            problem=problem,
            device=device,
            profiler=NullProfiler(),
            ctx=None,
            distributed_mode=False,
            image_size=[16, 16],
        )
        problem.resized.assert_called_once_with([16, 16], device=device)
        assert solver.problem is sentinel

    def test_denoiser_resizes_with_device(self):
        sentinel = _make_objective()
        problem = MagicMock()
        problem.resized.return_value = sentinel
        device = torch.device("cpu")
        solver = DenoiserSolver(
            problem=problem,
            device=device,
            profiler=NullProfiler(),
            ctx=None,
            distributed_mode=False,
            image_size=[16, 16],
        )
        problem.resized.assert_called_once_with([16, 16], device=device)
        assert solver.problem is sentinel


# ---------------------------------------------------------------------------
# UnrolledPnPSolver._setup_components / _setup_optimizer
# ---------------------------------------------------------------------------


class TestUnrolledPnPSolverSetupComponents:

    def test_returns_pgd_model_and_denoiser_params(self):
        from deepinv.optim import PGD

        solver = _make_unrolled_solver()
        with patch(
            "toolsbench.solver.unrolled_pnp.create_drunet_denoiser",
            return_value=_mock_denoiser(),
        ):
            model, denoiser_params = solver._setup_components()
        assert isinstance(model, PGD)
        assert isinstance(denoiser_params, list)

    def test_unknown_denoiser_raises(self):
        solver = _make_unrolled_solver(denoiser="unknown")
        with pytest.raises(ValueError, match="Unknown denoiser"):
            solver._setup_components()

    def test_setup_optimizer_with_algo_params(self):
        solver = _make_unrolled_solver(train_algo_params=True)
        with patch(
            "toolsbench.solver.unrolled_pnp.create_drunet_denoiser",
            return_value=_mock_denoiser(),
        ):
            _, denoiser_params = solver._setup_components()
        optimizer = solver._setup_optimizer(denoiser_params)
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2

    def test_setup_optimizer_without_algo_params(self):
        solver = _make_unrolled_solver(train_algo_params=False)
        with patch(
            "toolsbench.solver.unrolled_pnp.create_drunet_denoiser",
            return_value=_mock_denoiser(),
        ):
            _, denoiser_params = solver._setup_components()
        optimizer = solver._setup_optimizer(denoiser_params)
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 1


# ---------------------------------------------------------------------------
# UnrolledPnPSolver.get_result
# ---------------------------------------------------------------------------


class TestUnrolledPnPSolverGetResult:

    def test_includes_profiler_metrics(self):
        solver = _make_unrolled_solver()
        solver.reconstruction = torch.ones(1, 1, 4, 4)
        solver.profiler = MagicMock()
        solver.profiler.get_current_metrics.return_value = {
            "total_time_sec": 0.5,
            "max_gpu_mb": 100.0,
        }
        result = solver.get_result()
        assert result["total_time_sec"] == 0.5
        assert result["max_gpu_mb"] == 100.0

    def test_includes_ground_truth(self):
        solver = _make_unrolled_solver()
        solver.reconstruction = torch.ones(1, 1, 4, 4)
        solver.profiler = NullProfiler()
        result = solver.get_result()
        assert torch.equal(result["ground_truth"], solver.problem.ground_truth)

    def test_image_size_resizes_with_device(self):
        sentinel = _make_objective()
        problem = MagicMock()
        problem.resized.return_value = sentinel
        solver = _make_unrolled_solver(problem=problem, image_size=[16, 16])
        problem.resized.assert_called_once_with([16, 16], device=solver.device)
        assert solver.problem is sentinel


# ---------------------------------------------------------------------------
# DenoiserSolver
# ---------------------------------------------------------------------------


class TestDenoiserSolver:

    def test_compile_post_requires_distribute(self):
        with pytest.raises(ValueError, match="compile='post' requires"):
            DenoiserSolver(
                _make_objective(),
                torch.device("cpu"),
                NullProfiler(),
                None,
                False,
                compile="post",
                distribute_denoiser=False,
            )

    def test_get_result_includes_roofline_and_profiler(self):
        solver = DenoiserSolver.__new__(DenoiserSolver)
        solver.reconstruction = torch.ones(1, 1, 4, 4)
        solver.reference = torch.zeros(1, 1, 4, 4)
        solver.roofline_metrics = {
            "flops": 100,
            "mem_bytes": 10,
            "arith_intensity": 10.0,
        }
        solver.profiler = MagicMock()
        solver.profiler.get_current_metrics.return_value = {"denoise_time_sec": 0.5}

        result = solver.get_result()

        assert torch.equal(result["reconstruction"], torch.ones(1, 1, 4, 4))
        assert result["flops"] == 100
        assert result["arith_intensity"] == 10.0
        assert result["denoise_time_sec"] == 0.5


# ---------------------------------------------------------------------------
# EquivariantSolver — both presets share one interface, so one set covers both
# ---------------------------------------------------------------------------

EQUIVARIANT_VOLUME_SIZE = (8, 4, 8)


def _equivariant_setup(preset, **kwargs):
    """A cryo problem, its unsharded pair and the model the solver builds.

    Goes through ``_setup_model`` rather than constructing the network here, so
    the ``preset`` branch is what gets exercised.
    """
    problem = CryoEIInvProb().get_invprob(
        InvProbConfig(
            size=EQUIVARIANT_VOLUME_SIZE,
            batch_size=1,
            channels=1,
            device=torch.device("cpu"),
            params=dict(
                num_angles=7, noise_level=0.1, seed=0, tomography_backend="torch"
            ),
        )
    )
    pair = build_cryo_pair(
        problem.physics,
        problem.measurements,
        torch.device("cpu"),
        ctx=None,
        num_operators=None,
        backend="torch",
    )
    torch.manual_seed(0)
    solver = _make_equivariant_solver(
        problem=problem, preset=preset, f_maps=4, num_levels=2, **kwargs
    )
    model, module, _info = solver._setup_model(pair)
    return pair, solver, model, module


@pytest.mark.parametrize("preset", ["tomo_ei", "unrolled"])
class TestEquivariantSolver:
    def test_trains_two_steps(self, preset):
        """Two steps run: the loss stays finite and every trainable parameter
        receives a gradient. Under ``unrolled`` that set includes ``stepsize``,
        so no preset-specific assertion is needed."""
        pair, solver, model, module = _equivariant_setup(
            preset, n_iter=2, train_algo_params=True
        )
        trainable = [p for p in module.parameters() if p.requires_grad]
        assert trainable
        y_evn, y_odd = solver.problem.measurements
        obs = ObsLoss(gain="none", ramp=True)
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            x_net, y_net = solver._forward(model, pair, y_evn, y_odd)
            loss = obs(pair, x_net, y_net, y_evn, y_odd)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None for p in trainable)
            optimizer.step()

    def test_amp_keeps_physics_fp32(self, preset):
        """The denoiser runs reduced-precision; nothing downstream does.

        Both halves matter: without the dtype check on the denoiser, ``_amp``
        could be a no-op that only calls ``.float()`` and still pass. ``A`` is
        the thing that must never see bf16 — astra has no dtype guard.
        """
        pair, solver, model, module = _equivariant_setup(preset, mixed_precision="bf16")
        model = solver._amp(model)

        inner_dtypes = []
        conv = next(m for m in module.modules() if isinstance(m, torch.nn.Conv3d))
        handle = conv.register_forward_hook(
            lambda _m, _i, out: inner_dtypes.append(out.dtype)
        )
        seen = []
        operator = type(pair.physics_evn)
        original_A = operator.A
        operator.A = lambda self, x, *a, **k: (
            seen.append(x.dtype),
            original_A(self, x, *a, **k),
        )[1]
        try:
            y_evn, y_odd = solver.problem.measurements
            x_net, _ = solver._forward(model, pair, y_evn, y_odd)
        finally:
            operator.A = original_A
            handle.remove()

        assert inner_dtypes and set(inner_dtypes) == {torch.bfloat16}
        assert x_net.dtype == torch.float32
        if preset == "unrolled":
            # tomo_ei's forward never touches A; the PGD's does, every iteration.
            assert seen and set(seen) == {torch.float32}


def test_equivariant_mixed_precision_config():
    """The dtype map, the fp16-only scaler, and rejection of anything else."""
    for spelling, dtype in (
        ("off", None),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        solver = _make_equivariant_solver(mixed_precision=spelling)
        assert solver._amp_dtype is dtype
        # fp16 gradients underflow without a loss multiply; bf16 shares fp32's
        # exponent range, so it needs no scaler.
        assert (solver._scaler is not None) == (spelling == "fp16")
    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        _make_equivariant_solver(mixed_precision="fp8")


def test_clamp_stepsize():
    """Clamps a trainable stepsize, no-ops on a fixed one."""

    class _Model:
        def __init__(self, stepsize):
            self.params_algo = {"stepsize": stepsize}

    trainable = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(-0.5))])
    clamp_stepsize(_Model(trainable))
    assert float(trainable[0].detach()) > 0

    fixed = [-0.5]
    clamp_stepsize(_Model(fixed))
    assert fixed == [-0.5]
