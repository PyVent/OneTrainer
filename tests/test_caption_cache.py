import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from modules.dataLoader.ExpandCaptionSamples import ExpandCaptionSamples
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy

from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class _Paths(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ["image_path", "concept"]

    def length(self):
        return len(self.rows)

    def get_item(self, variation, index, requested_name=None):
        path, concept = self.rows[index]
        return {"image_path": path, "concept": concept}


class _Encoded(PipelineModule, RandomAccessPipelineModule):
    """Record which sample and augmentation a real cache asks an encoder to process."""

    def get_inputs(self):
        return ["image_path", "prompt"]

    def get_outputs(self):
        return ["encoded_image", "encoded_text"]

    def length(self):
        return self._get_previous_length("image_path")

    def get_item(self, variation, index, requested_name=None):
        return {
            "encoded_image": (self._get_previous_item(variation, "image_path", index), index, variation),
            "encoded_text": (self._get_previous_item(variation, "prompt", index), variation),
        }


class CaptionCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = TrainConfig.default_values()
        self.config.cache_dir = str(self.root / "cache")
        self.rows = []
        for number, names in enumerate((("a", "b"), ("c",), ("disabled",))):
            concept = ConceptConfig.default_values()
            concept.path = str(self.root)
            concept.seed = number
            concept.enabled = number != 2
            concept.text.caption_mode = "all" if number == 0 else "random"
            concept.balancing = 2 if number == 0 else 5
            concept.balancing_strategy = BalancingStrategy.REPEATS if number == 0 else BalancingStrategy.SAMPLES
            concept.image_variations = 2
            concept.text_variations = 1
            data = concept.to_dict()
            self.rows.extend((str(self.root / (name + ".png")), data) for name in names)
        (self.root / "a.txt").write_text("first caption\nsecond caption", encoding="utf-8")
        (self.root / "c.txt").write_text("one\ntwo\nthree", encoding="utf-8")

    def pipeline(self, cached, text_cached, epoch=0, index=0):
        self.config.latent_caching = cached
        names = ["image_path", "encoded_image", "encoded_text", "prompt", "concept"]
        caches = DataLoaderText2ImageMixin._cache_modules_from_names(
            None, Mock(), Mock(), ["encoded_image"], ["image_path"], ["encoded_text"], names,
            self.config, text_cached, before_cache_image_fun=lambda: None,
        )
        state = PipelineState(2)
        self.addCleanup(state.executor.shutdown, wait=True)
        return LoadingPipeline(
            torch.device("cpu"), [_Paths(self.rows), ExpandCaptionSamples(), _Encoded(), caches, OutputPipelineModule(names)],
            batch_size=1, seed=42, state=state, initial_epoch=epoch, initial_index=index,
        )

    @staticmethod
    def epoch(pipeline):
        pipeline.start_next_epoch()
        return list(pipeline)

    def test_repeats_mixed_caching_multiple_concepts_and_resume_keep_pairs(self):
        for cached, text_cached in ((False, False), (True, False), (True, True)):
            with self.subTest(cached=cached, text_cached=text_cached):
                pipeline = self.pipeline(cached, text_cached)
                epochs = [self.epoch(pipeline), self.epoch(pipeline)]
                for epoch, rows in enumerate(epochs):
                    self.assertEqual(len(rows), 11)
                    for position, row in enumerate(rows):
                        path, index, variation = row["encoded_image"]
                        self.assertEqual(path, row["image_path"])
                        self.assertEqual(row["encoded_text"][0], row["prompt"])
                        self.assertEqual(row["concept"]["seed"], 0 if position < 6 else 1)
                        expected_index = position % 3 if position < 6 else 3
                        self.assertEqual(index, expected_index)
                        requested_variation = epoch * 2 + position // 3 if position < 6 else epoch * 5 + position - 6
                        self.assertEqual(variation, requested_variation % 2 if cached else requested_variation)
                        self.assertEqual(row["encoded_text"][1], 0 if cached and text_cached else requested_variation)
                resumed = self.epoch(self.pipeline(cached, text_cached, epoch=1, index=2))
                self.assertEqual(resumed, epochs[1][2:])

    def test_caption_edit_refreshes_text_cache_without_reencoding_images(self):
        before = self.epoch(self.pipeline(True, True))
        (self.root / "a.txt").write_text("changed first\nchanged second", encoding="utf-8")
        after = self.epoch(self.pipeline(True, True))
        self.assertEqual([r["encoded_image"] for r in after], [r["encoded_image"] for r in before])
        self.assertEqual([r["prompt"] for r in after[:3]], ["changed first", "changed second", ""])
        self.assertEqual(after[0]["encoded_text"][0], "changed first")


if __name__ == "__main__":
    unittest.main()
