from pathlib import Path

import numpy as np
import torch
from benchopt import BaseDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset


class SimulatedStreamDataset(TorchDataset):
    """Dataset that loads cached image tensors and per-sample physics specs."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = torch.load(record["image_path"], map_location="cpu")
        return {
            "image": image,
            "image_path": record["image_path"],
            "physics_spec": dict(record["physics_spec"]),
        }


class Dataset(BaseDataset):
    """Synthetic stream dataset backed by on-disk images."""

    name = "single_image_blur_stream"
    requirements = [
        "pip::torch",
        "numpy",
    ]

    parameters = {
        "image_size": [256],
        "blur_sigma": [2.0],
        "noise_level": [0.01],
        "stream_length": [64],
        "rate_hz": [0.0],  # 0.0 -> as fast as possible
        "queue_capacity": [4],
        "drop_policy": ["block"],
        "seed": [42],
    }

    def get_data(self):
        device = torch.device("cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        records = self._build_stream_records(
            data_dir=self._stream_data_dir(),
            device=device,
        )
        stream_dataset = SimulatedStreamDataset(records)
        stream_dataloader = DataLoader(
            stream_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )
        if len(stream_dataset) == 0:
            raise ValueError("stream_length must be >= 1 to create a valid stream.")

        first_sample = stream_dataset[0]
        ground_truth = first_sample["image"].clone()
        physics_spec = dict(first_sample["physics_spec"])

        stream_spec = {
            "rate_hz": (None if self.rate_hz <= 0.0 else float(self.rate_hz)),
            "max_packets": len(stream_dataset),
            "queue_capacity": int(self.queue_capacity),
            "drop_policy": self.drop_policy,
            "include_ground_truth": True,
        }

        return dict(
            ground_truth=ground_truth,
            stream_dataloader=stream_dataloader,
            physics_spec=physics_spec,
            stream_spec=stream_spec,
            min_pixel=0.0,
            max_pixel=1.0,
        )

    def _stream_data_dir(self):
        root = Path(__file__).resolve().parents[1] / "data" / "simulated"
        return root / (
            f"size_{int(self.image_size)}_seed_{int(self.seed)}"
            f"_blur_{float(self.blur_sigma):.3f}_noise_{float(self.noise_level):.5f}"
        )

    def _build_stream_records(self, data_dir, device):
        data_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample_id in range(int(self.stream_length)):
            image_path = data_dir / f"image_{sample_id:06d}.pt"
            if not image_path.exists():
                image = self._generate_reference_image(
                    image_size=self.image_size,
                    device=device,
                    sample_id=sample_id,
                    seed=self.seed,
                ).cpu()
                torch.save(image, image_path)
            records.append(
                {
                    "image_path": str(image_path),
                    "physics_spec": {
                        "blur_sigma": float(self.blur_sigma),
                        "noise_level": float(self.noise_level),
                        "seed": int(self.seed),
                    },
                }
            )
        return records

    @staticmethod
    def _generate_reference_image(image_size, device, sample_id=0, seed=0):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, image_size, device=device),
            torch.linspace(-1.0, 1.0, image_size, device=device),
            indexing="ij",
        )
        radius = torch.sqrt(x**2 + y**2)
        shift_x = 0.25 * np.sin(0.37 * sample_id)
        shift_y = 0.25 * np.cos(0.23 * sample_id)
        image = 0.4 + 0.25 * torch.sin(8.0 * (x + shift_x)) * torch.cos(
            6.0 * (y + shift_y)
        )
        image += 0.35 * (radius < 0.55).float()
        image += (
            0.2
            * (
                ((x + 0.25 - 0.15 * shift_x) ** 2 + (y - 0.2 + 0.1 * shift_y) ** 2)
                < 0.08
            ).float()
        )
        noise_rng = torch.Generator(device=device).manual_seed(
            int(seed) + int(sample_id)
        )
        image += 0.03 * torch.rand(image.shape, generator=noise_rng, device=device)
        image = torch.clamp(image, 0.0, 1.0)
        return image.unsqueeze(0).unsqueeze(0)
