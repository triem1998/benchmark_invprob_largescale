import numpy as np
import torch
from benchopt import BaseDataset
from deepinv.physics import GaussianNoise
from deepinv.physics.blur import Blur, gaussian_blur


class Dataset(BaseDataset):
    """Single-image blur dataset with a stream specification."""

    name = "single_image_blur_stream"
    requirements = [
        "pip::torch",
        "numpy",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
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

        ground_truth = self._generate_reference_image(self.image_size, device=device)
        physics = self._build_blur_physics(device=device)
        measurement = physics(ground_truth)
        measurement = torch.clamp(measurement, 0.0, 1.0)

        stream_spec = {
            "rate_hz": (None if self.rate_hz <= 0.0 else float(self.rate_hz)),
            "max_packets": int(self.stream_length),
            "queue_capacity": int(self.queue_capacity),
            "drop_policy": self.drop_policy,
            "include_ground_truth": True,
        }

        return dict(
            ground_truth=ground_truth,
            physics=physics,
            measurement_template=measurement,
            physics_spec={
                "blur_sigma": float(self.blur_sigma),
                "noise_level": float(self.noise_level),
                "seed": int(self.seed),
            },
            stream_spec=stream_spec,
            min_pixel=0.0,
            max_pixel=1.0,
        )

    def _build_blur_physics(self, device):
        kernel = gaussian_blur(
            sigma=(self.blur_sigma, self.blur_sigma),
            angle=0.0,
            device=str(device),
        )
        physics = Blur(filter=kernel, padding="circular", device=str(device))
        rng = torch.Generator(device=device).manual_seed(self.seed)
        physics.noise_model = GaussianNoise(sigma=self.noise_level, rng=rng)
        return physics.to(device)

    @staticmethod
    def _generate_reference_image(image_size, device):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, image_size, device=device),
            torch.linspace(-1.0, 1.0, image_size, device=device),
            indexing="ij",
        )
        radius = torch.sqrt(x**2 + y**2)
        image = 0.4 + 0.25 * torch.sin(8.0 * x) * torch.cos(6.0 * y)
        image += 0.35 * (radius < 0.55).float()
        image += 0.2 * (((x + 0.25) ** 2 + (y - 0.2) ** 2) < 0.08).float()
        image = torch.clamp(image, 0.0, 1.0)
        return image.unsqueeze(0).unsqueeze(0)
