from __future__ import annotations

import builtins
import time
import typing

import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics.blur import Blur, gaussian_blur

# SimAI-Bench eagerly imports Dragon symbols. Define non-Dragon fallbacks.
if not hasattr(builtins, "Task"):
    builtins.Task = object
if not hasattr(builtins, "Any"):
    builtins.Any = typing.Any
if not hasattr(builtins, "Sequence"):
    builtins.Sequence = typing.Sequence

from SimAIBench import DataStore


class BoxBlurDenoiser(torch.nn.Module):
    """Simple denoiser used for a minimal PnP baseline."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x, sigma=None):
        del sigma
        if x.ndim == 4:
            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
        if x.ndim == 5:
            return torch.nn.functional.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
        return x


def _to_device(payload, device):
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    if isinstance(payload, list):
        return [_to_device(p, device) for p in payload]
    if isinstance(payload, tuple):
        return tuple(_to_device(p, device) for p in payload)
    return payload


def _concat_payloads(payloads):
    """Concatenate homogeneous payloads along batch dimension."""
    first = payloads[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(payloads, dim=0)
        return torch.cat(payloads, dim=0)
    if isinstance(first, list):
        return [_concat_payloads([p[i] for p in payloads]) for i in range(len(first))]
    if isinstance(first, tuple):
        return tuple(_concat_payloads([p[i] for p in payloads]) for i in range(len(first)))
    raise TypeError(f"Unsupported payload type for batching: {type(first)}")


def _run_pnp_updates(
    reconstruction,
    measurement,
    physics,
    data_fidelity,
    prior,
    pnp_cfg,
):
    with torch.no_grad():
        for _ in range(int(pnp_cfg["inner_iterations"])):
            grad = data_fidelity.grad(reconstruction, measurement, physics)
            reconstruction = reconstruction - pnp_cfg["step_size"] * grad
            denoised = prior.prox(
                reconstruction,
                sigma_denoiser=pnp_cfg["denoiser_sigma"],
            )
            if pnp_cfg["denoiser_lambda_relaxation"] is None:
                reconstruction = denoised
            else:
                lam = float(pnp_cfg["denoiser_lambda_relaxation"])
                alpha = (pnp_cfg["step_size"] * lam) / (1.0 + pnp_cfg["step_size"] * lam)
                reconstruction = (1.0 - alpha) * reconstruction + alpha * denoised
            reconstruction = reconstruction.clamp(
                pnp_cfg["min_pixel"],
                pnp_cfg["max_pixel"],
            )
    return reconstruction


def _clone_payload(payload):
    if isinstance(payload, torch.Tensor):
        return payload.detach().clone()
    if isinstance(payload, list):
        return [_clone_payload(x) for x in payload]
    if isinstance(payload, tuple):
        return tuple(_clone_payload(x) for x in payload)
    return payload


def payload_nbytes(payload) -> int:
    if isinstance(payload, torch.Tensor):
        return payload.nelement() * payload.element_size()
    if isinstance(payload, list):
        return sum(payload_nbytes(x) for x in payload)
    if isinstance(payload, tuple):
        return sum(payload_nbytes(x) for x in payload)
    return 0


def packet_key(key_prefix: str, packet_id: int) -> str:
    return f"{key_prefix}:packet:{packet_id:08d}"


def eos_key(key_prefix: str) -> str:
    return f"{key_prefix}:eos"


def result_key(key_prefix: str) -> str:
    return f"{key_prefix}:result"


def producer_component(
    server_info,
    key_prefix,
    measurement_template,
    ground_truth_template,
    stream_spec,
):
    """Workflow component: produce packet stream into SimAI DataStore."""
    ds = DataStore("producer", server_info=server_info)
    max_packets = int(stream_spec["max_packets"])
    rate_hz = stream_spec.get("rate_hz", None)
    include_ground_truth = bool(stream_spec.get("include_ground_truth", True))
    t0 = time.perf_counter()
    try:
        for packet_id in range(max_packets):
            if rate_hz is not None and float(rate_hz) > 0:
                target_t = t0 + (packet_id / float(rate_hz))
                wait_s = target_t - time.perf_counter()
                if wait_s > 0:
                    time.sleep(wait_s)

            y = _clone_payload(measurement_template)
            x_true = _clone_payload(ground_truth_template) if include_ground_truth else None
            ds.stage_write(
                packet_key(key_prefix, packet_id),
                {
                    "packet_id": packet_id,
                    "t_source": time.perf_counter(),
                    "y": y,
                    "x_true": x_true,
                    "nbytes": payload_nbytes(y),
                },
            )
    finally:
        ds.stage_write(eos_key(key_prefix), True)


def _build_physics_from_spec(ground_truth_shape, physics_spec, compute_device):
    blur_sigma = float(physics_spec["blur_sigma"])
    kernel = gaussian_blur(
        sigma=(blur_sigma, blur_sigma),
        angle=0.0,
        device=str(compute_device),
    )
    return Blur(filter=kernel, padding="circular", device=str(compute_device))


def pnp_consumer_component(
    server_info,
    key_prefix,
    physics_spec,
    ground_truth_shape,
    pnp_cfg,
):
    """Workflow component: consume packets, run PnP, and stage result."""
    ds = DataStore("consumer", server_info=server_info)

    if pnp_cfg["device"] == "cuda" and torch.cuda.is_available():
        compute_device = torch.device("cuda")
    else:
        compute_device = torch.device("cpu")

    physics = _build_physics_from_spec(
        ground_truth_shape=ground_truth_shape,
        physics_spec=physics_spec,
        compute_device=compute_device,
    )
    denoiser = BoxBlurDenoiser(kernel_size=pnp_cfg["denoiser_kernel_size"]).to(compute_device)
    prior = PnP(denoiser=denoiser)
    data_fidelity = L2()

    # Keep a single-image representative for objective quality reporting.
    reconstruction = torch.zeros(
        tuple(ground_truth_shape),
        device=compute_device,
        dtype=torch.float32,
    )
    consumed_packets = 0
    consumed_batches = 0
    consumed_bytes = 0
    first_consume_t = None
    last_consume_t = None
    packet_id = 0
    batch_size = max(1, int(pnp_cfg.get("batch_size", 1)))
    batch_wait_s = max(0.0, float(pnp_cfg.get("batch_wait_s", 0.0)))
    poll_interval_s = max(0.0, float(pnp_cfg["poll_interval_s"]))

    t_start = time.perf_counter()
    while True:
        key = packet_key(key_prefix, packet_id)
        if not ds.poll_staged_data(key):
            if ds.poll_staged_data(eos_key(key_prefix)):
                break
            time.sleep(poll_interval_s)
            continue

        # Start a batch with the first available packet, then gather more.
        batch_packets = [ds.stage_read(key)]
        packet_id += 1
        gather_start = time.perf_counter()
        while len(batch_packets) < batch_size:
            next_key = packet_key(key_prefix, packet_id)
            if ds.poll_staged_data(next_key):
                batch_packets.append(ds.stage_read(next_key))
                packet_id += 1
                continue
            if ds.poll_staged_data(eos_key(key_prefix)):
                break
            if time.perf_counter() - gather_start >= batch_wait_s:
                break
            time.sleep(poll_interval_s)

        measurement = _concat_payloads(
            [_to_device(packet["y"], compute_device) for packet in batch_packets]
        )

        if hasattr(physics, "A_adjoint"):
            with torch.no_grad():
                reconstruction_batch = physics.A_adjoint(measurement)
                reconstruction_batch = reconstruction_batch.clamp(
                    pnp_cfg["min_pixel"],
                    pnp_cfg["max_pixel"],
                )
        else:
            reconstruction_batch = torch.zeros_like(measurement)

        reconstruction_batch = _run_pnp_updates(
            reconstruction=reconstruction_batch,
            measurement=measurement,
            physics=physics,
            data_fidelity=data_fidelity,
            prior=prior,
            pnp_cfg=pnp_cfg,
        )
        reconstruction = reconstruction_batch[:1]

        now = time.perf_counter()
        if first_consume_t is None:
            first_consume_t = now
        last_consume_t = now
        consumed_packets += len(batch_packets)
        consumed_batches += 1
        consumed_bytes += sum(
            int(packet.get("nbytes", payload_nbytes(packet["y"])))
            for packet in batch_packets
        )

    t_end = time.perf_counter()
    ds.stage_write(
        result_key(key_prefix),
        {
            "reconstruction": reconstruction.detach().cpu(),
            "trace": {
                "t_start": t_start,
                "t_end": t_end,
                "first_consume_t": first_consume_t,
                "last_consume_t": last_consume_t,
                "consumed_packets": consumed_packets,
                "consumed_batches": consumed_batches,
                "consumed_bytes": consumed_bytes,
                "dropped_packets": 0,
            },
        },
    )
