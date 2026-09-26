"""Extend MGDS image loading with opt-in alpha preservation."""

from mgds.pipelineModules.LoadImage import LoadImage as MGDSLoadImage

import torch
from torchvision.io import ImageReadMode


class LoadImage(MGDSLoadImage):
    def __init__(
            self,
            path_in_name: str,
            image_out_name: str,
            range_min: float,
            range_max: float,
            supported_extensions: set[str],
            channels: int = 3,
            dtype: torch.dtype | None = None,
    ):
        # Retain MGDS's RGB/grayscale behavior, including its Pillow fallback.
        super().__init__(
            path_in_name, image_out_name, range_min, range_max, supported_extensions,
            channels=3 if channels == 4 else channels, dtype=dtype,
        )
        if channels == 4:
            # Both readers preserve existing alpha and make RGB inputs opaque.
            self.mode = ImageReadMode.RGB_ALPHA
            self.pillow_mode = "RGBA"
