"""Diffusers pipeline for Anima with the Qwen Image 2.1 VAE."""

import torch

from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

import numpy as np
from PIL import Image


class Anima21Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->text_conditioner->transformer->vae"

    def __init__(self, scheduler, t5_tokenizer, text_conditioner, text_encoder, tokenizer, transformer, vae):
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            t5_tokenizer=t5_tokenizer,
            text_conditioner=text_conditioner,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
        )

    def _encode_prompt(self, prompt: str, device):
        qwen = self.tokenizer([prompt], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        t5 = self.t5_tokenizer([prompt], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        ids = qwen.input_ids.to(device)
        mask = qwen.attention_mask.to(device)
        hidden = self.text_encoder(ids, attention_mask=mask.float()).last_hidden_state
        hidden = hidden * mask.to(hidden).unsqueeze(-1)
        return self.text_conditioner(
            source_hidden_states=hidden.to(self.text_conditioner.dtype),
            target_input_ids=t5.input_ids.to(device),
            target_attention_mask=t5.attention_mask.to(device),
            source_attention_mask=mask,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        vae_scale_factor = self.vae.spatial_compression_ratio
        patch_height, patch_width = self.transformer.config.patch_size[-2:]
        if height <= 0 or width <= 0 or height % (vae_scale_factor * patch_height) or width % (vae_scale_factor * patch_width):
            raise ValueError(
                f"Height and width must be positive multiples of "
                f"{vae_scale_factor * patch_height} and {vae_scale_factor * patch_width}"
            )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        device = self._execution_device
        prompts = [prompt, negative_prompt] if guidance_scale > 1 else [prompt]
        embeddings = torch.cat([self._encode_prompt(value, device) for value in prompts])
        latents = torch.randn(
            (1, self.vae.config.z_dim, 1, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        padding_mask = torch.zeros((1, 1, height, width), device=device, dtype=self.transformer.dtype)
        self.scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps), device=device
        )
        for timestep in self.scheduler.timesteps:
            predicted = self.transformer(
                hidden_states=latents.repeat(len(prompts), 1, 1, 1, 1).to(self.transformer.dtype),
                timestep=(timestep / self.scheduler.config.num_train_timesteps).expand(len(prompts)),
                encoder_hidden_states=embeddings.to(self.transformer.dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            if len(prompts) == 2:
                positive, negative = predicted.chunk(2)
                predicted = negative + guidance_scale * (positive - negative)
            latents = self.scheduler.step(predicted.float(), timestep, latents, return_dict=False)[0]

        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=self.vae.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device, dtype=self.vae.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        decoded = self.vae.decode(latents.to(self.vae.dtype) * std + mean, return_dict=False)[0][0, :, 0]
        array = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(array)
        self.maybe_free_model_hooks()
        if not return_dict:
            return ([image],)
        return ImagePipelineOutput(images=[image])
