from pathlib import Path

from modules.model.AnimaModel import AnimaModel

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule

import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm


class SaveAnimaDebug(PipelineModule, SingleVariationRandomAccessPipelineModule):
    """Save aligned prompts, decoded images and masks before serial batch sorting."""

    def __init__(self, model: AnimaModel, path: str, masked: bool):
        super().__init__()
        self.model = model
        self.path = Path(path)
        self.masked = masked

    def get_inputs(self) -> list[str]:
        return ["image_path", "latent_image", "prompt"] + (["latent_mask"] if self.masked else [])

    def get_outputs(self) -> list[str]:
        return []

    def length(self) -> int:
        return 0

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {}

    @torch.no_grad()
    def start(self, variation: int):
        directory = self.path / f"epoch-{variation}"
        directory.mkdir(parents=True, exist_ok=True)
        self.model.materialize("vae")
        vae = self.model.vae
        for index in tqdm(range(self._get_previous_length("image_path")), desc="writing Anima debug images"):
            image_path = self._get_previous_item(variation, "image_path", index)
            prefix = f"{index}-{Path(image_path).stem}"
            latent = self._get_previous_item(variation, "latent_image", index).to(device=vae.device, dtype=vae.dtype)
            with self.model.autocast_context:
                decoded = vae.decode(latent.unsqueeze(0)).sample[0, :, 0].float()
            pixels = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            # Always PNG: the input may be JPEG even when this VAE produces RGBA.
            Image.fromarray(pixels).save(directory / f"{prefix}-decoded_image.png")

            if self.masked:
                mask = self._get_previous_item(variation, "latent_mask", index).float()
                mask = F.interpolate(mask[:, 0].unsqueeze(0), size=decoded.shape[-2:], mode="nearest")[0, 0]
                pixels = (mask.clamp(0, 1) * 255).round().byte().cpu().numpy()
                Image.fromarray(pixels).save(directory / f"{prefix}-decoded_mask.png")

            prompt = self._get_previous_item(variation, "prompt", index)
            (directory / f"{prefix}-prompt.txt").write_text(prompt, encoding="utf-8")
