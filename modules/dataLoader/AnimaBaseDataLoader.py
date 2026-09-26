import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.EncodeAnimaVAE import EncodeAnimaVAE
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.SaveAnimaDebug import SaveAnimaDebug
from modules.model.AnimaModel import PROMPT_MAX_LENGTH, AnimaModel
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.EncodeAnimaText import EncodeAnimaText
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


@factory.register(BaseDataLoader, ModelType.ANIMA_QWEN21_VAE)
@factory.register(BaseDataLoader, ModelType.ANIMA)
class AnimaBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: AnimaModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeAnimaVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        vae_scale_factor = model.vae.spatial_compression_ratio
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=1 / vae_scale_factor)
        # Anima has no chat template — tokenize raw prompt with both tokenizers
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.tokenizer, max_token_length=PROMPT_MAX_LENGTH)
        tokenize_t5 = Tokenize(in_name='prompt', tokens_out_name='t5_tokens', mask_out_name='t5_tokens_mask', tokenizer=model.t5_tokenizer, max_token_length=PROMPT_MAX_LENGTH)
        # EncodeAnimaText runs Qwen3 encoder + AnimaTextConditioner; output is fixed (512, 1024)
        encode_prompt = EncodeAnimaText(
            tokens_name='tokens', tokens_attention_mask_name='tokens_mask',
            t5_tokens_name='t5_tokens', t5_tokens_attention_mask_name='t5_tokens_mask',
            hidden_state_out_name='text_encoder_hidden_state',
            text_encoder=model.text_encoder, text_conditioner=model.text_conditioner,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules += [tokenize_prompt, tokenize_t5, encode_prompt]

        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = ['tokens', 'tokens_mask', 't5_tokens', 't5_tokens_mask', 'text_encoder_hidden_state']

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 't5_tokens', 't5_tokens_mask', 'text_encoder_hidden_state',
            'concept'
        ]

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=True,
            # Invalidate caches produced before RGBA support and keep the original Anima cache separate.
            cache_namespace=f"anima-qwen21-v2-{model.image_channels}ch"
                if model.model_type == ModelType.ANIMA_QWEN21_VAE else None,
        )

    def _output_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
            'tokens_mask',
            't5_tokens',
            't5_tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_hidden_state')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: AnimaModel):
        return [SaveAnimaDebug(
            model, os.path.join(config.debug_dir, "dataloader"),
            masked=config.masked_training or config.model_type.has_mask_input(),
        )]

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(self,
            config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=64,
            allow_video_files=False, #don't allow video files, but...
            vae_frame_dim=True,  #...Anima has a video-capable VAE. convert images to video dimensions
        )
