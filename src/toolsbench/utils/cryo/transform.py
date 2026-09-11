"""Random 90°-multiple 3D rotation for the equivariance loss.

The 40-element ``k_set`` is
icecream's, kept verbatim so the EI loss sees the same distribution of
augmentations; the deepinv ``Transform`` base class is dropped, since the two
methods below are all the loss calls.
"""

from __future__ import annotations

import torch

__all__ = ["Rotate3D"]


class Rotate3D:
    """Random rotation drawn from the cubic symmetry group (plus flips).

    :param tuple[int,int,int] | None volume_shape: when given and not a cube,
        only rotations preserving this shape are sampled — the astra geometry
        is bound to a fixed ``volume_shape``, so a rotation that transposes two
        axes of a non-cubic volume cannot be projected. ``None`` keeps all 40.
    """

    _KSET = [
        [0, 0, 1, -1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 3, -1],
        [0, 0, 3, 1],
        [0, 1, 0, -1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 2],
        [0, 1, 1, -1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, -1],
        [0, 1, 2, 1],
        [0, 1, 3, -1],
        [0, 1, 3, 1],
        [0, 2, 1, -1],
        [0, 2, 3, -1],
        [0, 3, 0, -1],
        [0, 3, 1, -1],
        [0, 3, 2, -1],
        [0, 3, 3, -1],
        [1, 0, 0, -1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 2],
        [1, 0, 1, -1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 2],
        [1, 0, 2, -1],
        [1, 0, 2, 1],
        [1, 0, 3, -1],
        [1, 0, 3, 1],
        [1, 2, 0, -1],
        [1, 2, 1, -1],
        [1, 2, 2, -1],
        [1, 2, 3, -1],
    ]

    def __init__(self, volume_shape: tuple[int, int, int] | None = None) -> None:
        self.valid_indices = self._shape_preserving_indices(volume_shape)

    @staticmethod
    def _rotated_shape(shape, kx: int, ky: int, kz: int) -> tuple[int, int, int]:
        """Shape after the same rot90 sequence ``transform`` applies — an odd
        count swaps the two axes it acts on; flips never change shape."""
        s = list(shape)
        for k, (a, b) in ((kx, (1, 2)), (ky, (0, 2)), (kz, (0, 1))):
            if k % 2:
                s[a], s[b] = s[b], s[a]
        return tuple(s)

    @classmethod
    def _shape_preserving_indices(cls, volume_shape) -> list[int]:
        if volume_shape is None:
            return list(range(len(cls._KSET)))
        shape = tuple(int(s) for s in volume_shape)
        return [
            i
            for i, (kx, ky, kz, _axis) in enumerate(cls._KSET)
            if cls._rotated_shape(shape, kx, ky, kz) == shape
        ]

    def get_params(self, x: torch.Tensor) -> dict:
        """Sample one rotation index, shared by the whole batch."""
        idx = int(torch.randint(len(self.valid_indices), (1,)).item())
        return {"k_idx": self.valid_indices[idx]}

    def transform(self, x: torch.Tensor, k_idx: int = 0, **kwargs) -> torch.Tensor:
        """Apply rotation ``k_idx`` to ``x`` of shape (B, C, D, H, W)."""
        kx, ky, kz, axis = self._KSET[k_idx]
        out = torch.rot90(x, k=kx, dims=(3, 4))
        out = torch.rot90(out, k=ky, dims=(2, 4))
        out = torch.rot90(out, k=kz, dims=(2, 3))
        if axis != -1:
            out = torch.flip(out, [axis + 2])
        return out
