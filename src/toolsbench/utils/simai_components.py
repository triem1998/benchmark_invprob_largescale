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
        return tuple(
            _concat_payloads([p[i] for p in payloads]) for i in range(len(first))
        )
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
                alpha = (pnp_cfg["step_size"] * lam) / (
                    1.0 + pnp_cfg["step_size"] * lam
                )
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


def _normalize_image_tensor(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected image tensor, got {type(image)}")
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")
    return image.to(dtype=torch.float32)


def _sample_to_image_and_spec(sample):
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dictionary, got {type(sample)}")

    image = sample.get("image", None)
    image_path = sample.get("image_path", None)
    physics_spec = sample.get("physics_spec", None)

    if image is None and image_path is not None:
        image = torch.load(image_path, map_location="cpu")
    if image is None:
        raise KeyError("Missing image payload. Expected 'image' or 'image_path'.")
    if physics_spec is None:
        raise KeyError("Missing 'physics_spec' in stream sample.")

    return _normalize_image_tensor(image), dict(physics_spec)


def producer_component(
    server_info,
    key_prefix,
    stream_spec,
    stream_records=None,
    stream_dataloader=None,
):
    """Workflow component: produce packet stream into SimAI DataStore."""
    ds = DataStore("producer", server_info=server_info)
    max_packets = int(stream_spec["max_packets"])
    rate_hz = stream_spec.get("rate_hz", None)
    include_ground_truth = bool(stream_spec.get("include_ground_truth", True))
    t0 = time.perf_counter()
    current_physics_spec = None
    current_physics = None
    if stream_records is not None:
        source_iter = iter(stream_records)
    elif stream_dataloader is not None:
        source_iter = iter(stream_dataloader)
    else:
        raise ValueError(
            "producer_component expects either stream_records or stream_dataloader."
        )
    try:
        for packet_id, sample in enumerate(source_iter):
            if packet_id >= max_packets:
                break
            if rate_hz is not None and float(rate_hz) > 0:
                target_t = t0 + (packet_id / float(rate_hz))
                wait_s = target_t - time.perf_counter()
                if wait_s > 0:
                    time.sleep(wait_s)

            x_true_raw, sample_physics_spec = _sample_to_image_and_spec(sample)

            if sample_physics_spec != current_physics_spec or current_physics is None:
                current_physics = _build_physics_from_spec(
                    ground_truth_shape=tuple(x_true_raw.shape),
                    physics_spec=sample_physics_spec,
                    compute_device=torch.device("cpu"),
                )
                current_physics_spec = dict(sample_physics_spec)

            with torch.no_grad():
                y = current_physics(x_true_raw)
                y = torch.clamp(y, 0.0, 1.0)

            x_true = _clone_payload(x_true_raw) if include_ground_truth else None
            ds.stage_write(
                packet_key(key_prefix, packet_id),
                {
                    "packet_id": packet_id,
                    "t_source": time.perf_counter(),
                    "y": _clone_payload(y),
                    "x_true": x_true,
                    "physics_spec": sample_physics_spec,
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
    active_physics_spec = dict(physics_spec)
    denoiser = BoxBlurDenoiser(kernel_size=pnp_cfg["denoiser_kernel_size"]).to(
        compute_device
    )
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
        batch_physics_spec = batch_packets[0].get("physics_spec", active_physics_spec)
        if batch_physics_spec != active_physics_spec:
            physics = _build_physics_from_spec(
                ground_truth_shape=ground_truth_shape,
                physics_spec=batch_physics_spec,
                compute_device=compute_device,
            )
            active_physics_spec = dict(batch_physics_spec)

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
