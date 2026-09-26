import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.modelSaver.pixartAlpha.PixArtAlphaEmbeddingSaver import PixArtAlphaEmbeddingSaver
from modules.modelSaver.sana.SanaEmbeddingSaver import SanaEmbeddingSaver
from modules.modelSaver.stableDiffusion.StableDiffusionEmbeddingSaver import StableDiffusionEmbeddingSaver
from modules.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import StableDiffusion3EmbeddingSaver
from modules.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import StableDiffusionXLEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import load_file


def component(value, uuid="main", placeholder="<test token>"):
    return SimpleNamespace(
        uuid=uuid, placeholder=placeholder,
        vector=torch.full((2, 3), float(value), requires_grad=True),
        output_vector=torch.full((2, 3), float(value + 10), requires_grad=True),
    )


def model_with(embedding=None, states=None, additional=()):
    return SimpleNamespace(
        embedding=embedding, embedding_state_dicts=states or {}, additional_embeddings=list(additional),
    )


class EmbeddingSaverTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.destination = str(self.directory / "embedding.safetensors")

    def test_all_model_formats_keep_encoder_keys_and_values(self):
        single = SimpleNamespace(text_encoder_embedding=component(1))
        multi = SimpleNamespace(**{f"text_encoder_{i}_embedding": component(i) for i in range(1, 5)})
        prior = SimpleNamespace(prior_text_encoder_embedding=component(1))
        cases = (
            (ChromaEmbeddingSaver, single, ("t5",)),
            (FluxEmbeddingSaver, multi, ("clip_l", "t5")),
            (HiDreamEmbeddingSaver, multi, ("clip_l", "clip_g", "t5", "llama")),
            (HunyuanVideoEmbeddingSaver, multi, ("llama", "clip_l")),
            (PixArtAlphaEmbeddingSaver, single, ("t5",)),
            (SanaEmbeddingSaver, single, ("gemma",)),
            (StableDiffusionEmbeddingSaver, single, ("emp_params",)),
            (StableDiffusion3EmbeddingSaver, multi, ("clip_l", "clip_g", "t5")),
            (StableDiffusionXLEmbeddingSaver, multi, ("clip_l", "clip_g")),
            (WuerstchenEmbeddingSaver, prior, ("clip_g",)),
        )
        for saver_class, embedding, keys in cases:
            with self.subTest(saver=saver_class.__name__):
                saver_class().save_single(
                    model_with(embedding), ModelFormat.SAFETENSORS, self.destination, torch.float16,
                )
                saved = load_file(self.destination)
                self.assertEqual(set(saved), set(keys) | {f"{key}_out" for key in keys})
                for i, key in enumerate(keys, 1):
                    torch.testing.assert_close(saved[key], torch.full((2, 3), float(i), dtype=torch.float16))
                    torch.testing.assert_close(saved[f"{key}_out"], torch.full((2, 3), float(i + 10), dtype=torch.float16))

    def test_live_embedding_uses_matching_cached_state_for_missing_encoder(self):
        embedding = SimpleNamespace(text_encoder_1_embedding=component(1), text_encoder_2_embedding=component(2))
        embedding.text_encoder_2_embedding.vector = None
        embedding.text_encoder_2_embedding.output_vector = None
        cached = torch.arange(6, dtype=torch.float32).reshape(3, 2).T
        model = model_with(embedding, {"other": {"t5": torch.zeros(1)}, "main": {"t5": cached}})
        FluxEmbeddingSaver().save_single(model, ModelFormat.SAFETENSORS, self.destination, torch.float16)
        saved = load_file(self.destination)
        torch.testing.assert_close(saved["t5"], cached.to(torch.float16))
        self.assertTrue(saved["t5"].is_contiguous())
        self.assertIs(model.embedding_state_dicts["main"]["t5"], cached)
        self.assertEqual(cached.dtype, torch.float32)
        self.assertEqual(embedding.text_encoder_1_embedding.vector.dtype, torch.float32)
        self.assertTrue(embedding.text_encoder_1_embedding.vector.requires_grad)

    def test_raw_embedding_conversion_needs_no_live_embedding(self):
        cached = torch.arange(6, dtype=torch.float32).reshape(3, 2).T
        model = model_with(states={"main": {"emp_params": cached}})
        StableDiffusionEmbeddingSaver().save_single(model, ModelFormat.SAFETENSORS, self.destination, torch.bfloat16)
        saved = load_file(self.destination)
        torch.testing.assert_close(saved["emp_params"], cached.to(torch.bfloat16))

    def test_additional_embeddings_support_live_and_cached_only_entries(self):
        primary = SimpleNamespace(text_encoder_embedding=component(1))
        additional = SimpleNamespace(text_encoder_embedding=component(2, uuid="extra", placeholder="<extra token>"))
        cached = torch.full((2, 3), 5.0)
        model = model_with(primary, {"main": {"t5": cached}, "cached": {"t5": cached}}, [additional])
        ChromaEmbeddingSaver().save_multiple(model, ModelFormat.SAFETENSORS, self.destination, torch.float16)
        folder = Path(self.destination + "_embeddings")
        self.assertEqual({p.name for p in folder.iterdir()}, {"extra_token.safetensors", "cached.safetensors"})
        torch.testing.assert_close(load_file(str(folder / "cached.safetensors"))["t5"], cached.to(torch.float16))
        self.assertTrue(torch.all(load_file(str(folder / "extra_token.safetensors"))["t5"] == 2))

    def test_internal_backup_uses_uuid_and_preserves_original_precision(self):
        embedding = SimpleNamespace(text_encoder_embedding=component(1))
        model = model_with(embedding, {"cached": {"t5": torch.ones(2, 3, dtype=torch.bfloat16)}})
        saver = ChromaEmbeddingSaver()
        saver.save_multiple(model, ModelFormat.INTERNAL, str(self.directory), torch.float16)
        saver.save_single(model, ModelFormat.INTERNAL, str(self.directory), torch.float16)
        folder = self.directory / "embeddings"
        self.assertEqual({p.name for p in folder.iterdir()}, {"main.safetensors", "cached.safetensors"})
        self.assertEqual(load_file(str(folder / "main.safetensors"))["t5"].dtype, torch.float32)
        self.assertEqual(load_file(str(folder / "cached.safetensors"))["t5"].dtype, torch.bfloat16)

    def test_empty_single_save_has_actionable_error_and_creates_no_file(self):
        with self.assertRaisesRegex(ValueError, "No embedding"):
            ChromaEmbeddingSaver().save_single(model_with(), ModelFormat.SAFETENSORS, self.destination, None)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_model_without_additional_embeddings_needs_no_embedding_output_format(self):
        ChromaEmbeddingSaver().save_multiple(model_with(), ModelFormat.DIFFUSERS, self.destination, None)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
