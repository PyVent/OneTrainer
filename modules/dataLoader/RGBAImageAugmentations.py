"""Color augmentations for MGDS CHW images and CTHW videos, preserving alpha."""

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch
from torchvision.transforms import functional


class _ColorAugmentation(PipelineModule, RandomAccessPipelineModule):
    hue = False

    def __init__(self, names, enabled_in_name, fixed_enabled_in_name, max_strength_in_name):
        super().__init__()
        self.names = names
        self.enabled_in_name = enabled_in_name
        self.fixed_enabled_in_name = fixed_enabled_in_name
        self.max_strength_in_name = max_strength_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.enabled_in_name, self.fixed_enabled_in_name, self.max_strength_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        fixed_enabled = self._get_previous_item(variation, self.fixed_enabled_in_name, index)
        max_strength = self._get_previous_item(variation, self.max_strength_in_name, index)
        rand = self._get_rand(variation, index)
        if self.hue:
            strength = max(-0.5, min(0.5, rand.uniform(-max_strength * 0.5, max_strength * 0.5))) \
                if enabled else max_strength * 0.5
        else:
            strength = max(0.0, rand.uniform(1 - max_strength, 1 + max_strength)) \
                if enabled else 1.0 + max_strength

        item = {}
        for name in self.names:
            image = self._get_previous_item(variation, name, index)
            if image is not None and (enabled or fixed_enabled):
                # torchvision expects (..., C, H, W); MGDS stores video as (C, T, H, W).
                image = image.movedim(0, -3)
                if image.shape[-3] == 4:
                    rgb = self.operation(image[..., :3, :, :], strength)
                    image = torch.cat((rgb, image[..., 3:, :, :]), dim=-3)
                else:
                    image = self.operation(image, strength)
                image = image.movedim(-3, 0)
            item[name] = image
        return item


class RandomBrightness(_ColorAugmentation):
    operation = staticmethod(functional.adjust_brightness)


class RandomContrast(_ColorAugmentation):
    operation = staticmethod(functional.adjust_contrast)


class RandomSaturation(_ColorAugmentation):
    operation = staticmethod(functional.adjust_saturation)


class RandomHue(_ColorAugmentation):
    hue = True
    operation = staticmethod(functional.adjust_hue)
