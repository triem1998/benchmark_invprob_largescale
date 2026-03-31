from __future__ import annotations

import builtins
import os
import shutil
import tempfile
import time
import typing

import torch
from benchopt import BaseSolver

# SimAI-Bench currently imports Dragon executor symbols eagerly.
# Define fallbacks so non-Dragon installs can still import high-level APIs.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence

from SimAIBench import DataStore, OchestratorConfig, ServerManager, SystemConfig, Workflow
from SimAIBench import server_registry

from toolsbench.utils.simai_components import (
    pnp_consumer_component,
    producer_component,
    result_key,
)


class Solver(BaseSolver):
    """Single-process PnP solver using SimAI-Bench high-level APIs."""

    name = "PnPStreamSimAIBench"
    requirements = [
        "pip::torch",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
        "SimAIBench",
    ]
    sampling_strategy = "run_once"

    parameters = {
        "step_size": [0.8],
        "denoiser_sigma": [0.05],
        "denoiser_kernel_size": [3],
        "denoiser_lambda_relaxation": [None],
        "inner_iterations": [1],
        "batch_size": [1, 4],
        "batch_wait_s": [0.0],
        "device": ["cpu"],
        "poll_interval_s": [0.01],
    }

    def set_objective(
        self,
        physics,
        measurement_template,
        physics_spec,
        stream_spec,
        ground_truth_shape,
        min_pixel=0.0,
        max_pixel=1.0,
    ):
        self.physics = physics
        self.measurement_template = measurement_template
        self.physics_spec = physics_spec
        self.stream_spec = stream_spec
        self.ground_truth_shape = tuple(ground_truth_shape)
        self.min_pixel = float(min_pixel)
        self.max_pixel = float(max_pixel)

        self.reconstruction = torch.zeros(self.ground_truth_shape, dtype=torch.float32)
        self.trace = {}
        self.name = self.__class__.name

    def _build_workflow(self, server_info, key_prefix, compute_device):
        pnp_cfg = {
            "step_size": float(self.step_size),
            "denoiser_sigma": float(self.denoiser_sigma),
            "denoiser_kernel_size": int(self.denoiser_kernel_size),
            "denoiser_lambda_relaxation": self.denoiser_lambda_relaxation,
            "inner_iterations": int(self.inner_iterations),
            "batch_size": int(self.batch_size),
            "batch_wait_s": float(self.batch_wait_s),
            "min_pixel": float(self.min_pixel),
            "max_pixel": float(self.max_pixel),
            "device": str(compute_device),
            "poll_interval_s": float(self.poll_interval_s),
        }

        workflow = Workflow(
            orchestrator_config=OchestratorConfig(
                name="process-pool",
                submit_loop_sleep_time=1,
            ),
            system_config=SystemConfig(name="local", ncpus=2, ngpus=0),
        )
        workflow.register_component(
            name="producer",
            executable=producer_component,
            type="local",
            args={
                "server_info": server_info,
                "key_prefix": key_prefix,
                "measurement_template": self.measurement_template,
                "ground_truth_template": None,
                "stream_spec": self.stream_spec,
            },
        )
        workflow.register_component(
            name="pnp_consumer",
            executable=pnp_consumer_component,
            type="local",
            args={
                "server_info": server_info,
                "key_prefix": key_prefix,
                "physics_spec": self.physics_spec,
                "ground_truth_shape": self.ground_truth_shape,
                "pnp_cfg": pnp_cfg,
            },
        )
        return workflow

    @staticmethod
    def _wait_for_result(result_reader, key, timeout_s=30.0):
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            if result_reader.poll_staged_data(key):
                return result_reader.stage_read(key)
            time.sleep(0.1)
        raise RuntimeError(
            f"Timed out waiting for workflow result key '{key}'. "
            "Check SimAI-Bench logs for component failures."
        )

    def run(self, n_iter=None):
        del n_iter
        compute_device = (
            torch.device("cuda")
            if self.device == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        tmp_dir = tempfile.mkdtemp(prefix="infer_inverse_simai_")
        server_config = server_registry.create_config(
            type="filesystem",
            server_address=tmp_dir,
            nshards=64,
        )
        server = ServerManager("stream_server", config=server_config)
        server.start_server()
        try:
            server_info = server.get_server_info()
            key_prefix = f"run_{time.time_ns()}"

            workflow = self._build_workflow(
                server_info=server_info,
                key_prefix=key_prefix,
                compute_device=compute_device,
            )
            workflow.launch()

            result_reader = DataStore("result_reader", server_info=server_info)
            output = self._wait_for_result(
                result_reader=result_reader,
                key=result_key(key_prefix),
                timeout_s=30.0,
            )

            self.reconstruction = output["reconstruction"]
            self.trace = output["trace"]
        finally:
            server.stop_server()
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def get_result(self):
        return dict(
            reconstruction=self.reconstruction,
            trace=self.trace,
            name=self.name,
        )
