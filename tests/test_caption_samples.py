import tempfile
import unittest
from pathlib import Path
from random import Random
from unittest.mock import patch

from modules.dataLoader.AnimaBaseDataLoader import AnimaBaseDataLoader
from modules.dataLoader.ExpandCaptionSamples import ExpandCaptionSamples
from modules.modelSetup.AnimaFineTuneSetup import AnimaFineTuneSetup
from modules.util.caption_util import caption_format, caption_key, load_captions, read_caption_lines
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.TrainProgress import TrainProgress

import torch

from PIL import Image
from test_anima_qwen21_vae import MODEL_TYPES, small_model


class CaptionSamplesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.concept = ConceptConfig.default_values()
        self.concept.path = str(self.root)

    def expand(self, paths, concepts=None):
        module = ExpandCaptionSamples()
        concepts = concepts or [self.concept.to_dict()] * len(paths)
        module._get_previous_length = lambda _: len(paths)
        module._get_previous_item = lambda _, name, index: paths[index] if name == "image_path" else concepts[index]
        module._get_rand = lambda variation, index: Random(variation * 100 + index)
        module.start(0)
        return module

    def test_old_config_keeps_legacy_selection_and_tag_rules(self):
        restored = ConceptConfig.default_values().from_dict({"__version": 2, "text": {"enable_tag_shuffling": True}})
        self.assertEqual(restored.text.caption_mode, "random")
        self.assertEqual(restored.text.caption_format, "tags")
        self.assertTrue(restored.text.enable_tag_shuffling)

    def test_all_lines_are_stable_distinct_examples_and_missing_is_retained(self):
        (self.root / "a.txt").write_text("\ufeffred, blue, green\n\nA blue house in the forest.\n", encoding="utf-8")
        self.concept.text.caption_mode = "all"
        self.concept.text.caption_format = "auto"
        module = self.expand([str(self.root / "a.png"), str(self.root / "missing.png")])
        self.assertEqual(module.length(), 3)
        for epoch in range(3):
            rows = [module.get_item(epoch, index) for index in range(3)]
            self.assertEqual([row["prompt"] for row in rows], ["red, blue, green", "A blue house in the forest.", ""])
            self.assertEqual(rows[0]["image_path"], rows[1]["image_path"])
            self.assertEqual([row["caption_is_tags"] for row in rows], [True, False, False])

    def test_random_mode_does_not_expand_shared_or_per_image_captions(self):
        shared = self.root / "a.txt"
        shared.write_text("one\ntwo\nthree", encoding="utf-8")
        for source in ("sample", "concept"):
            self.concept.text.prompt_source = source
            self.concept.text.prompt_path = str(shared)
            module = self.expand([str(self.root / "a.png")])
            self.assertEqual(module.length(), 1)
            self.assertEqual({module.get_item(epoch, 0)["prompt"] for epoch in range(50)}, {"one", "two", "three"})

    def test_random_indexing_does_not_load_all_caption_text(self):
        with patch("modules.dataLoader.ExpandCaptionSamples.read_caption_lines") as read:
            module = self.expand([str(self.root / "a.png")])
            module.get_item(0, 0, "concept")
            read.assert_not_called()

    def test_shared_captions_expand_each_image_and_filename_stays_single(self):
        shared = self.root / "shared.txt"
        shared.write_text("один\nдва", encoding="utf-8")
        self.concept.text.caption_mode = "all"
        self.concept.text.prompt_source = "concept"
        self.concept.text.prompt_path = str(shared)
        paths = [str(self.root / "a.png"), str(self.root / "b.png")]
        module = self.expand(paths)
        self.assertEqual(module.length(), 4)
        self.assertEqual([module.get_item(0, i)["image_path"] for i in range(4)], [paths[0]] * 2 + [paths[1]] * 2)
        self.concept.text.prompt_source = "filename"
        module = self.expand(paths)
        self.assertEqual([module.get_item(0, i)["prompt"] for i in range(2)], ["a", "b"])

    def test_cache_identity_tracks_contents_and_expanded_sample_mapping(self):
        caption = self.root / "a.txt"
        caption.write_text("one\ntwo", encoding="utf-8")
        self.concept.text.caption_mode = "all"
        paths = [str(self.root / "a.png")]
        before = self.expand(paths).get_item(0, 0)["concept"]
        caption.write_text("one\nchanged", encoding="utf-8")
        edited = self.expand(paths).get_item(0, 0)["concept"]
        self.assertEqual(before["_image_sample_key"], edited["_image_sample_key"])
        self.assertNotEqual(before["_text_sample_key"], edited["_text_sample_key"])
        caption.write_text("one\nchanged\nthird", encoding="utf-8")
        expanded = self.expand(paths).get_item(0, 0)["concept"]
        self.assertNotEqual(edited["_image_sample_key"], expanded["_image_sample_key"])
        self.assertNotIn("_image_sample_key", self.concept.to_dict())

    def test_format_override_is_content_based_and_roundtrips(self):
        self.concept.text.caption_format = "auto"
        ambiguous = "red, blue"
        self.assertEqual(caption_format(ambiguous, self.concept.text.to_dict()), "text")
        self.concept.text.caption_overrides[caption_key(ambiguous)] = "tags"
        config = ConceptConfig.default_values().from_dict(self.concept.to_dict())
        (self.root / "a.txt").write_text("A description.\n" + ambiguous, encoding="utf-8")
        captions = load_captions(self.root / "a.png", config.text.to_dict())
        self.assertEqual(caption_format(captions[1], config.text.to_dict()), "tags")
        self.assertEqual(caption_format("a red car, a blue sky, a green tree", config.text.to_dict()), "text")

    def test_empty_and_invalid_utf8(self):
        path = self.root / "caption.txt"
        path.write_text(" \n\n", encoding="utf-8")
        self.assertEqual(read_caption_lines(path), [""])
        path.write_bytes(b"\xff\xfe\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8.*caption.txt"):
            read_caption_lines(path)

    def test_real_anima_pipeline_separates_captions_augmentations_and_cache(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        for model_type in MODEL_TYPES:
            for cached in (False, True):
                with self.subTest(model=model_type, cached=cached), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    model = small_model(model_type, root)
                    config = model.train_config
                    data = root / "data"
                    data.mkdir()
                    Image.new("RGBA", (64, 64), (70, 80, 90, 128)).save(data / "sample.png")
                    Image.new("L", (64, 64), 255).save(data / "sample-masklabel.png")
                    prose = "A test image with three colors."
                    (data / "sample.txt").write_text("test, blue, green\n" + prose, encoding="utf-8")
                    concept = ConceptConfig.default_values()
                    concept.path = str(data)
                    concept.text.caption_mode = "all"
                    concept.text.caption_format = "auto"
                    concept.text.tag_dropout_enable = True
                    concept.text.tag_dropout_probability = 1.0
                    concept.text.keep_tags_count = 1
                    concept.image.enable_random_brightness = True
                    concept.image.random_brightness_max_strength = 0.5
                    concept.balancing = 2.0
                    config.concepts = [concept]
                    config.resolution = "64"
                    config.aspect_ratio_bucketing = False
                    config.batch_size = 1
                    config.masked_training = True
                    config.latent_caching = cached
                    config.dataloader_threads = 2
                    config.cache_dir = str(root / "cache")
                    setup = AnimaFineTuneSetup(torch.device("cpu"), torch.device("cpu"), False)
                    loader = AnimaBaseDataLoader(torch.device("cpu"), torch.device("cpu"), config, model, setup, TrainProgress())
                    epochs = []
                    for _ in range(2):
                        loader.get_data_set().start_next_epoch()
                        rows = list(loader.get_data_loader())
                        self.assertEqual(len(rows), 4)
                        self.assertCountEqual([row["prompt"][0] for row in rows], ["test", prose] * 2)
                        for row in rows:
                            expected = model.tokenizer(row["prompt"][0], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                            torch.testing.assert_close(row["tokens"], expected.input_ids)
                            self.assertTrue(torch.isfinite(row["latent_image"]).all())
                            self.assertIn("latent_mask", row)
                        epochs.append(rows)
                    by_prompt = {text: [row["latent_image"] for row in epochs[0] if row["prompt"][0] == text] for text in ("test", prose)}
                    self.assertFalse(torch.equal(by_prompt["test"][0], by_prompt[prose][0]))
                    if cached:
                        torch.testing.assert_close(by_prompt["test"][0], by_prompt["test"][1])
                    else:
                        self.assertFalse(torch.equal(by_prompt["test"][0], by_prompt["test"][1]))


if __name__ == "__main__":
    unittest.main()
