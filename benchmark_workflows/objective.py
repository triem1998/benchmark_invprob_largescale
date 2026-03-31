import math

from benchopt import BaseObjective

from toolsbench.utils import compute_psnr


class Objective(BaseObjective):
    """Reconstruction objective with objective-side throughput evaluation."""

    name = "streaming_reconstruction_objective"
    requirements = []
    min_benchopt_version = "1.8"
    sampling_strategy = "run_once"

    def set_data(
        self,
        ground_truth,
        physics,
        measurement_template,
        physics_spec,
        stream_spec,
        min_pixel=0.0,
        max_pixel=1.0,
    ):
        self.ground_truth = ground_truth
        self.physics = physics
        self.measurement_template = measurement_template
        self.physics_spec = physics_spec
        self.stream_spec = stream_spec
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def get_objective(self):
        return dict(
            physics=self.physics,
            measurement_template=self.measurement_template,
            physics_spec=self.physics_spec,
            stream_spec=self.stream_spec,
            ground_truth_shape=tuple(self.ground_truth.shape),
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
        )

    def evaluate_result(self, reconstruction, trace, name):
        del name
        psnr = compute_psnr(
            reconstruction=reconstruction,
            reference=self.ground_truth,
            max_pixel=self.max_pixel,
        )

        consumed_packets = int(trace.get("consumed_packets", 0))
        consumed_batches = int(trace.get("consumed_batches", 0))
        consumed_bytes = int(trace.get("consumed_bytes", 0))
        dropped_packets = int(trace.get("dropped_packets", 0))

        first_consume_t = trace.get("first_consume_t")
        last_consume_t = trace.get("last_consume_t")
        total_runtime_s = max(0.0, float(trace["t_end"]) - float(trace["t_start"]))

        if (
            first_consume_t is not None
            and last_consume_t is not None
            and float(last_consume_t) > float(first_consume_t)
        ):
            throughput_window_s = float(last_consume_t) - float(first_consume_t)
        else:
            throughput_window_s = total_runtime_s

        if throughput_window_s > 0.0 and consumed_packets > 0:
            throughput_fps = consumed_packets / throughput_window_s
            throughput_mb_s = consumed_bytes / throughput_window_s / (1024.0**2)
        else:
            throughput_fps = math.nan
            throughput_mb_s = math.nan
        if consumed_batches > 0:
            avg_batch_size = consumed_packets / consumed_batches
        else:
            avg_batch_size = math.nan

        return dict(
            value=-psnr,
            psnr=psnr,
            throughput_fps=throughput_fps,
            throughput_mb_s=throughput_mb_s,
            total_runtime_s=total_runtime_s,
            throughput_window_s=throughput_window_s,
            consumed_packets=consumed_packets,
            consumed_batches=consumed_batches,
            avg_batch_size=avg_batch_size,
            consumed_bytes=consumed_bytes,
            dropped_packets=dropped_packets,
        )

    def get_one_result(self):
        return dict(
            reconstruction=self.ground_truth.clone(),
            trace={
                "t_start": 0.0,
                "t_end": 1.0,
                "first_consume_t": 0.0,
                "last_consume_t": 1.0,
                "consumed_packets": 1,
                "consumed_batches": 1,
                "consumed_bytes": self.ground_truth.nelement()
                * self.ground_truth.element_size(),
                "dropped_packets": 0,
            },
            name="dummy",
        )
