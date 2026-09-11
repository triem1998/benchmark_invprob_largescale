"""The IceCream ``UNet3D`` and the wrapper that adapts it to deepinv.

The 3D network this case needs; other benchmark cases use deepinv's 2D
denoisers through ``toolsbench.utils.create_denoiser``.
"""

from __future__ import annotations

import torch
from deepinv.models.base import Denoiser

from .unet3d import UNet3D

__all__ = ["IceCreamUNetWrapper", "UNet3D", "build_unet3d"]

#: Architecture defaults for this case.
UNET_F_MAPS = 64
UNET_NUM_LEVELS = 4


class IceCreamUNetWrapper(Denoiser):
    """Wraps ``UNet3D`` so ``model(x, physics)`` works.

    Subclassing ``deepinv.models.base.Denoiser`` is not cosmetic: deepinv's
    tiling only engages when the target of ``distribute(..., "denoiser")`` is a
    ``Denoiser``.
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.unet(x)


def build_unet3d(
    device,
    f_maps: int = UNET_F_MAPS,
    num_levels: int = UNET_NUM_LEVELS,
    dropout: float = 0.0,
) -> tuple[torch.nn.Module, str]:
    """The wrapped UNet3D on ``device``.

    ``UNet3D`` only inserts Dropout when ``layer_order`` asks ('d');
    ``dropout_prob`` alone is inert.
    """
    layer_order = "crd" if dropout > 0 else "cr"
    unet = UNet3D(
        in_channels=1,
        out_channels=1,
        f_maps=f_maps,
        num_levels=num_levels,
        layer_order=layer_order,
        use_bias=False,
        dropout_prob=dropout,
    ).to(device)
    info = f"unet3d f_maps={f_maps} num_levels={num_levels} layer_order={layer_order}"
    return IceCreamUNetWrapper(unet).to(device), info
