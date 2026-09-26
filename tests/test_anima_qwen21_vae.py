import gc
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from unittest.mock import patch

from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader
from modules.dataLoader.EncodeAnimaVAE import EncodeAnimaVAE
from modules.model.anima.custom_vae import AutoencoderKLQwenImage21
from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.AnimaModelLoader import AnimaFineTuneModelLoader, AnimaLoRALoader, AnimaModelLoader
from modules.modelSampler.AnimaSampler import AnimaSampler
from modules.modelSaver.anima.AnimaModelSaver import AnimaModelSaver
from modules.modelSaver.AnimaFineTuneModelSaver import AnimaFineTuneModelSaver
from modules.modelSaver.AnimaLoRAModelSaver import AnimaLoRAModelSaver
from modules.modelSetup.AnimaFineTuneSetup import AnimaFineTuneSetup
from modules.modelSetup.AnimaLoRASetup import AnimaLoRASetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.TrainProgress import TrainProgress

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import Unigram

MODEL_TYPES = (ModelType.ANIMA, ModelType.ANIMA_QWEN21_VAE)


def small_vae(model_type):
    is_qwen21 = model_type == ModelType.ANIMA_QWEN21_VAE
    channels = 64 if is_qwen21 else 16
    args = {
        "base_dim": 8, "z_dim": channels, "num_res_blocks": 1,
        "dim_mult": [1] * (5 if is_qwen21 else 4),
        "temperal_downsample": [False] * (4 if is_qwen21 else 3),
        "latents_mean": [0.25] * channels, "latents_std": [1.5] * channels,
    }
    return AutoencoderKLQwenImage21(decoder_base_dim=8, **args) if is_qwen21 else AutoencoderKLQwenImage(**args)


def small_model(model_type, directory):
    model = AnimaModel(model_type)
    model.train_config = TrainConfig.default_values()
    config = model.train_config
    config.model_type = model_type
    config.train_device = config.temp_device = "cpu"
    config.train_dtype = config.fallback_train_dtype = DataType.FLOAT_32
    config.text_encoder.train = False
    config.text_encoder.weight_dtype = config.transformer.weight_dtype = config.vae.weight_dtype = DataType.FLOAT_32
    model.vae = small_vae(model_type).requires_grad_(False).eval()
    is_qwen21 = model_type == ModelType.ANIMA_QWEN21_VAE
    model.transformer = CosmosTransformer3DModel(
        in_channels=model.vae.config.z_dim, out_channels=model.vae.config.z_dim,
        num_attention_heads=2, attention_head_dim=32, num_layers=1, mlp_ratio=2,
        text_embed_dim=32, encoder_hidden_states_channels=32, adaln_lora_dim=8,
        patch_size=(1, 1, 1) if is_qwen21 else (1, 2, 2), extra_pos_embed_type=None,
        max_size=(8, 64, 64),
    )
    vocab_path, merges_path = directory / "vocab.json", directory / "merges.txt"
    vocab_path.write_text(json.dumps({"<|endoftext|>": 0, "t": 1, "e": 2, "s": 3}), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    model.tokenizer = Qwen2Tokenizer(str(vocab_path), str(merges_path))
    tokenizer = Tokenizer(Unigram([("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("test", -1.0)], unk_id=2))
    model.t5_tokenizer = T5TokenizerFast(tokenizer_object=tokenizer, extra_ids=0)
    model.text_encoder = Qwen3Model(Qwen3Config(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    )).requires_grad_(False).eval()
    model.text_conditioner = AnimaTextConditioner(
        source_dim=32, target_dim=32, model_dim=32, num_layers=1, num_attention_heads=2,
        target_vocab_size=len(model.t5_tokenizer), min_sequence_length=4,
    ).requires_grad_(False).eval()
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler()
    return model


class AnimaVaeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_encode_decode_and_latent_scaling_for_both_variants(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type), torch.no_grad():
                model = AnimaModel(model_type)
                model.vae = small_vae(model_type)
                image = torch.rand(1, model.image_channels, 1, 32, 64) * 2 - 1
                latents = model.vae.encode(image).latent_dist.mode()
                scale = 16 if model_type == ModelType.ANIMA_QWEN21_VAE else 8
                channels = 64 if model_type == ModelType.ANIMA_QWEN21_VAE else 16
                self.assertEqual(tuple(latents.shape), (1, channels, 1, 32 // scale, 64 // scale))
                torch.testing.assert_close(model.unscale_latents(model.scale_latents(latents)), latents)
                decoded = model.vae.decode(latents).sample
                self.assertEqual(decoded.shape, image.shape)
                self.assertTrue(torch.isfinite(decoded).all())

    def test_parallel_encoding_is_consistent_for_both_variants(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type):
                model = AnimaModel(model_type)
                model.vae = small_vae(model_type).requires_grad_(False)
                images = [torch.rand(model.image_channels, 1, 32, 32) for _ in range(3)]
                encoder = EncodeAnimaVAE("image", "latent", model.vae)
                encoder._get_previous_item = lambda _variation, _name, index, images=images: images[index]
                expected = [encoder.get_item(0, i)["latent"].mode() for i in range(3)]
                with ThreadPoolExecutor(max_workers=3) as executor:
                    actual = list(executor.map(lambda i, encoder=encoder: encoder.get_item(0, i)["latent"].mode(), range(3)))
                for sequential, concurrent in zip(expected, actual, strict=True):
                    torch.testing.assert_close(concurrent, sequential)

    def test_loader_rejects_wrong_vae_type_before_loading_weights(self):
        for selected, stored_class in (
            (ModelType.ANIMA, "AutoencoderKLQwenImage21"),
            (ModelType.ANIMA_QWEN21_VAE, "AutoencoderKLQwenImage"),
        ):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                vae_path = Path(directory) / "vae"
                vae_path.mkdir()
                (vae_path / "config.json").write_text(json.dumps({"_class_name": stored_class}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Select the matching Anima model type") as caught:
                    AnimaModelLoader().load(AnimaModel(selected), selected, ModelNames(base_model=directory), None, None)
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_diffusers_save_load_and_internal_resume_for_both_variants(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = small_model(model_type, root)
                destination = root / "checkpoint"
                # Exercise the dtype copy used by real model exports, including both tokenizers.
                AnimaModelSaver().save(model, ModelFormat.DIFFUSERS, str(destination), torch.float32)
                if model_type == ModelType.ANIMA_QWEN21_VAE:
                    index = json.loads((destination / "model_index.json").read_text(encoding="utf-8"))
                    self.assertEqual(index["_class_name"], ["pipeline_anima21", "Anima21Pipeline"])
                    self.assertEqual(index["vae"], ["custom_vae", "AutoencoderKLQwenImage21"])
                    exported = DiffusionPipeline.from_pretrained(str(destination), local_files_only=True, trust_remote_code=True)
                    result = exported("test", height=32, width=32, num_inference_steps=1)
                    self.assertEqual(result.images[0].mode, "RGBA")
                    del exported
                    gc.collect()
                    # OneTrainer must load without ANY Python files in the model directory.
                    for source in destination.rglob("*.py"):
                        source.unlink()
                else:
                    self.assertFalse((destination / "custom_vae.py").exists())
                config = model.train_config
                restored = AnimaFineTuneModelLoader().load(
                    model_type, ModelNames(base_model=str(destination)), config.weight_dtypes(), config.quantization,
                )
                self.assertEqual(type(restored.vae), type(model.vae))
                self.assertEqual(restored.image_channels, model.image_channels)
                for name, weight in model.vae.state_dict().items():
                    torch.testing.assert_close(restored.vae.state_dict()[name], weight, rtol=0, atol=0)
                for name, weight in model.transformer.state_dict().items():
                    torch.testing.assert_close(restored.transformer.state_dict()[name], weight, rtol=0, atol=0)

                transformer_path = root / "transformer.safetensors"
                AnimaModelSaver().save(model, ModelFormat.ORIGINAL_TRANSFORMER, str(transformer_path), torch.float32)
                transformer_restored = AnimaFineTuneModelLoader().load(
                    model_type, ModelNames(base_model=str(destination), transformer_model=str(transformer_path)),
                    config.weight_dtypes(), config.quantization,
                )
                for name, weight in model.transformer.state_dict().items():
                    torch.testing.assert_close(transformer_restored.transformer.state_dict()[name], weight, rtol=0, atol=0)

                model.optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=1e-4)
                model.param_group_mapping = ["transformer"]
                model.ema = None
                model.train_progress = TrainProgress(epoch=2, epoch_step=3, epoch_sample=4, global_step=5)
                backup = root / "backup"
                AnimaFineTuneModelSaver().save(model, model_type, ModelFormat.INTERNAL, str(backup), None)
                resumed = AnimaFineTuneModelLoader().load(
                    model_type, ModelNames(base_model=str(backup)), config.weight_dtypes(), config.quantization,
                )
                self.assertEqual(resumed.train_progress.global_step, 5)
                self.assertEqual(resumed.train_progress.epoch, 2)
                self.assertIsNotNone(resumed.optimizer_state_dict)

    def test_mgds_training_and_sampling_with_rgb_and_rgba_inputs(self):
        for model_type in MODEL_TYPES:
            for cached in (False, True):
                with self.subTest(model_type=model_type, cached=cached), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    model = small_model(model_type, root)
                    config = model.train_config
                    dataset_path = root / "images"
                    dataset_path.mkdir()
                    Image.new("RGB", (64, 64), (100, 150, 200)).save(dataset_path / "rgb.jpg")
                    Image.new("RGBA", (64, 64), (50, 100, 200, 85)).save(dataset_path / "rgba.png")
                    for filename in ("rgb", "rgba"):
                        (dataset_path / f"{filename}.txt").write_text(f"test {filename} прозрачность", encoding="utf-8")
                        Image.new("L", (64, 64), 255).save(dataset_path / f"{filename}-masklabel.png")
                    concept = ConceptConfig.default_values()
                    concept.path = str(dataset_path)
                    concept.image.enable_fixed_hue = True
                    concept.image.random_hue_max_strength = 0.2
                    config.concepts = [concept]
                    config.resolution = "64"
                    config.aspect_ratio_bucketing = False
                    config.batch_size = 1
                    config.masked_training = True
                    config.latent_caching = cached
                    config.dataloader_threads = 2
                    config.cache_dir = str(root / "cache")
                    config.debug_mode = True
                    config.debug_dir = str(root / "debug")
                    config.text_encoder.dropout_probability = 0.5
                    setup = AnimaFineTuneSetup(torch.device("cpu"), torch.device("cpu"), False)
                    setup.setup_train_device(model, config)
                    loader = AnimaBaseDataLoader(
                        torch.device("cpu"), torch.device("cpu"), config, model, setup, TrainProgress(),
                    )
                    loader.get_data_set().start_next_epoch()
                    batches = list(loader.get_data_loader())
                    self.assertEqual(len(batches), 2)
                    self._check_debug_exports(root, model, 0)
                    loader.get_data_set().start_next_epoch()
                    next_batches = list(loader.get_data_loader())
                    self.assertEqual({x["image_path"][0] for x in batches},
                                     {x["image_path"][0] for x in next_batches})
                    self._check_debug_exports(root, model, 1)
                    batch = batches[0]
                    scale = model.vae.spatial_compression_ratio
                    self.assertEqual(tuple(batch["latent_image"].shape), (1, model.vae.config.z_dim, 1, 64 // scale, 64 // scale))
                    self.assertEqual(tuple(batch["latent_mask"].shape), (1, 1, 1, 64 // scale, 64 // scale))
                    optimizer = torch.optim.SGD(model.transformer.parameters(), lr=0.01)
                    before = model.transformer.proj_out.weight.detach().clone()
                    prediction = setup.predict(model, batch, config, TrainProgress())
                    loss = setup.calculate_loss(model, batch, prediction, config)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertIsNotNone(model.transformer.proj_out.weight.grad)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, model.transformer.proj_out.weight))

                    sampler = AnimaSampler(torch.device("cpu"), torch.device("cpu"), model, model_type)
                    result = sampler._AnimaSampler__sample_base(
                        prompt="test", negative_prompt="", height=64, width=64, seed=1,
                        random_seed=False, diffusion_steps=1, cfg_scale=2, noise_scheduler=None,
                    )
                    self.assertEqual(result.data.mode, "RGBA" if model.image_channels == 4 else "RGB")
                    self.assertEqual(result.data.size, (64, 64))

    def _check_debug_exports(self, root, model, epoch):
        folder = root / "debug" / "dataloader" / f"epoch-{epoch}"
        images = list(folder.glob("*-decoded_image.png"))
        self.assertEqual(len(images), 2)
        for image in images:
            prefix = image.name.removesuffix("-decoded_image.png")
            name = prefix.split("-", 1)[1]
            self.assertEqual((folder / f"{prefix}-prompt.txt").read_text(encoding="utf-8"),
                             f"test {name} прозрачность")
            with Image.open(image) as decoded, Image.open(folder / f"{prefix}-decoded_mask.png") as mask:
                self.assertEqual(decoded.mode, "RGBA" if model.image_channels == 4 else "RGB")
                self.assertEqual(decoded.size, (64, 64))
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.size, decoded.size)
                self.assertEqual(mask.getextrema(), (255, 255))

    def test_caption_dropout_uses_empty_prompt_for_cached_and_live_text(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                model = small_model(model_type, Path(directory))
                device = torch.device("cpu")
                positive = model.encode_text(device, text=["test"] * 3)
                original = positive.clone()
                empty = model.encode_text(device, text="")
                for probability, selected in ((0.0, []), (1.0, [0, 1, 2]), (0.5, [0])):
                    expected = positive.clone()
                    expected[selected] = empty
                    live = model.encode_text(
                        device, text=["test"] * 3, rand=Random(1),
                        text_encoder_dropout_probability=probability,
                    )
                    cached = model.encode_text(
                        device, text_encoder_output=positive, rand=Random(1),
                        text_encoder_dropout_probability=probability,
                    )
                    torch.testing.assert_close(live, expected)
                    torch.testing.assert_close(cached, expected)
                    torch.testing.assert_close(positive, original)
                for invalid in (-0.1, 1.1, float("nan")):
                    with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                        model.encode_text(device, text_encoder_output=positive, text_encoder_dropout_probability=invalid)

    def test_dropout_conditioning_survives_text_encoder_offloading_for_both_setups(self):
        for setup_type in (AnimaFineTuneSetup, AnimaLoRASetup):
            with self.subTest(setup=setup_type), tempfile.TemporaryDirectory() as directory:
                model = small_model(ModelType.ANIMA_QWEN21_VAE, Path(directory))
                config = model.train_config
                config.latent_caching = True
                config.text_encoder.dropout_probability = 1.0
                setup = setup_type(torch.device("cpu"), torch.device("cpu"), False)
                setup.setup_train_device(model, config)
                empty = model.empty_text_encoder_output
                self.assertIsNotNone(empty)
                self.assertFalse(empty.requires_grad)
                positive = torch.ones_like(empty, dtype=torch.float16).expand(3, -1, -1)
                with patch.object(model.text_encoder, "forward", side_effect=AssertionError("Encoder is offloaded")):
                    dropped = model.encode_text(
                        torch.device("cpu"), text_encoder_output=positive, text_encoder_dropout_probability=1.0,
                    )
                    setup.setup_train_device(model, config)
                torch.testing.assert_close(dropped, empty.to(positive).expand_as(positive))

    def test_uncached_text_encoding_accepts_dataset_tokens(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            model = small_model(ModelType.ANIMA_QWEN21_VAE, Path(directory))
            qwen = model.tokenizer("test", return_tensors="pt")
            t5 = model.t5_tokenizer("test", return_tensors="pt")
            encoded = model.encode_text(
                torch.device("cpu"), tokens=qwen.input_ids, tokens_mask=qwen.attention_mask,
                t5_tokens=t5.input_ids, t5_tokens_mask=t5.attention_mask,
            )
            self.assertEqual(encoded.shape[-1], 32)
            self.assertTrue(torch.isfinite(encoded).all())

    def test_lora_export_and_reload_for_both_variants(self):
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = small_model(model_type, root)
                config = model.train_config
                config.training_method = TrainingMethod.LORA
                config.layer_filter = "attn1,attn2,ff,patch_embed.proj,proj_out"
                model.transformer_lora = LoRAModuleWrapper(model.transformer, "transformer", config, config.layer_filter.split(","))
                original = model.transformer_lora.state_dict()
                # Nonzero updates ensure the test checks adapter behavior, including folded alpha in PEFT exports.
                for name, weight in original.items():
                    if name.endswith(".lora_up.weight"):
                        weight.normal_(mean=0.0, std=0.01)
                self.assertTrue(any("patch_embed" in key for key in original))
                self.assertTrue(any("proj_out" in key for key in original))
                for output_format in model_type.supported_lora_formats():
                    with self.subTest(output_format=output_format):
                        path = root / f"{output_format}.safetensors"
                        AnimaLoRAModelSaver().save(model, model_type, output_format, str(path), torch.float32)
                        restored = AnimaModel(model_type)
                        restored.transformer = model.transformer
                        AnimaLoRALoader().load(restored, ModelNames(lora=str(path)))
                        self.assertEqual(set(restored.lora_state_dict), set(original))
                        for name in original:
                            if name.endswith(".lora_up.weight"):
                                prefix = name.removesuffix(".lora_up.weight")
                                updates = []
                                for state in (original, restored.lora_state_dict):
                                    down = state[prefix + ".lora_down.weight"]
                                    up = state[prefix + ".lora_up.weight"]
                                    updates.append((up @ down) * (state[prefix + ".alpha"] / down.shape[0]))
                                torch.testing.assert_close(updates[1], updates[0])

    def test_factory_and_config_keep_anima_variants_distinct(self):
        from modules.util import create

        for model_type in MODEL_TYPES:
            config = TrainConfig.default_values()
            config.model_type = model_type
            restored = TrainConfig.default_values().from_dict(config.to_dict())
            self.assertEqual(restored.model_type, model_type)
            for method in (TrainingMethod.FINE_TUNE, TrainingMethod.LORA):
                self.assertIsNotNone(create.create_model_loader(model_type, method))
                self.assertIsNotNone(create.create_model_saver(model_type, method))
                self.assertIsNotNone(create.get_model_setup_class(model_type, method))

    def test_cache_paths_separate_model_variants(self):
        from mgds.pipelineModules.DiskCache import DiskCache

        paths = []
        for model_type in MODEL_TYPES:
            model = AnimaModel(model_type)
            model.vae = small_vae(model_type)
            config = TrainConfig.default_values()
            config.model_type = model_type
            config.cache_dir = "shared-cache"
            config.latent_caching = True
            loader = object.__new__(AnimaBaseDataLoader)
            modules = loader._cache_modules(config, model, None)
            paths.append([module.cache_dir for module in modules if isinstance(module, DiskCache)])
        self.assertTrue(set(paths[0]).isdisjoint(paths[1]))
        self.assertEqual(Path(paths[0][0]), Path("shared-cache/image"))
        self.assertEqual(Path(paths[1][0]).parent, Path("shared-cache/image"))


if __name__ == "__main__":
    unittest.main()
