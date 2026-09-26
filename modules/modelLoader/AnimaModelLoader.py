import math

from modules.model.anima.custom_vae import AutoencoderKLQwenImage21
from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        vae_type = AutoencoderKLQwenImage21 if model_type == ModelType.ANIMA_QWEN21_VAE else AutoencoderKLQwenImage
        vae_config = vae_type.load_config(
            vae_model_name or base_model_name, subfolder=None if vae_model_name else "vae",
        )
        declared_class = vae_config.get("_class_name")
        if declared_class and declared_class != vae_type.__name__:
            raise ValueError(
                f"{model_type} requires {vae_type.__name__}, but the selected VAE is {declared_class}. "
                "Select the matching Anima model type and checkpoint."
            )
        image_channels = vae_config.get("in_channels", 4) if model_type == ModelType.ANIMA_QWEN21_VAE \
            else vae_config.get("input_channels", 3)
        allowed_channels = (3, 4) if model_type == ModelType.ANIMA_QWEN21_VAE else (3,)
        if image_channels not in allowed_channels or vae_config.get("out_channels", image_channels) != image_channels:
            raise ValueError(f"Unsupported Anima VAE image channels: {image_channels}; input and output must match")

        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        if vae_model_name: #TODO simplify
            vae = self._load_diffusers_sub_module(
                vae_type,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                vae_type,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        if transformer.config.in_channels != vae.config.z_dim:
            raise ValueError(
                f"Anima transformer expects {transformer.config.in_channels} latent channels, "
                f"but the VAE produces {vae.config.z_dim}"
            )
        if transformer.config.out_channels != vae.config.z_dim:
            raise ValueError(
                f"Anima transformer produces {transformer.config.out_channels} latent channels, "
                f"but the VAE expects {vae.config.z_dim}"
            )
        if len(vae.config.latents_mean) != vae.config.z_dim or len(vae.config.latents_std) != vae.config.z_dim \
                or not all(math.isfinite(x) for x in vae.config.latents_mean) \
                or not all(math.isfinite(x) and x > 0 for x in vae.config.latents_std):
            raise ValueError("Anima VAE latent normalization must provide a finite mean and positive std per channel")

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.vae = vae
        model.transformer = transformer

    def load( #TODO share code between models
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        # Internal backups also store Diffusers components; the generic loader restores training metadata.
        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception as exc:
            raise RuntimeError(f"could not load model: {model_names.base_model}: {exc}") from exc


class AnimaLoRALoader(
    LoRALoaderMixin,
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json",
        ModelType.ANIMA_QWEN21_VAE: "resources/sd_model_spec/anima-qwen21-vae-lora.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=AnimaLoRALoader,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.ANIMA: "resources/sd_model_spec/anima.json",
        ModelType.ANIMA_QWEN21_VAE: "resources/sd_model_spec/anima-qwen21-vae.json",
    },
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
