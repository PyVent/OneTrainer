import importlib
import io
import pickle
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import patch

from modules.dataLoader.LoadImage import LoadImage
from modules.dataLoader.RGBAImageAugmentations import RandomBrightness, RandomContrast, RandomHue, RandomSaturation
from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType

from mgds.pipelineModules.LoadImage import LoadImage as MGDSLoadImage

import torch
from torchvision.transforms import functional

from PIL import Image


class RGBAImageTest(unittest.TestCase):
    def load_image(self, path, channels, loader_class=LoadImage, dtype=torch.float32):
        loader = loader_class("path", "image", 0, 1, {".png", ".webp", ".jpg"}, channels, dtype)
        loader.pipeline = SimpleNamespace(device=torch.device("cpu"))
        loader._get_previous_item = lambda *_: str(path)
        return loader.get_item(0, 0)["image"]

    def test_alpha_is_preserved_and_rgb_gets_opaque_alpha_for_all_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, color in (("RGB", (20, 40, 60)), ("RGBA", (20, 40, 60, 85)), ("LA", (30, 85))):
                source = Image.new(mode, (4, 3), color)
                for extension in ("png", "webp"):
                    path = root / f"{mode}.{extension}"
                    source.save(path, lossless=True)
                    for pillow in (False, True):
                        with self.subTest(mode=mode, extension=extension, pillow=pillow):
                            context = patch("mgds.pipelineModules.LoadImage.read_image", side_effect=RuntimeError()) \
                                if pillow else nullcontext()
                            with context:
                                tensor = self.load_image(path, 4)
                            self.assertEqual(tuple(tensor.shape), (4, 3, 4))
                            expected_alpha = 1.0 if mode == "RGB" else 85 / 255
                            torch.testing.assert_close(tensor[3], torch.full((3, 4), expected_alpha))

    def test_palette_transparency_and_jpeg_are_loaded_as_rgba(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("P", (3, 2), 0)
            image.putpalette([255, 0, 0] + [0] * 765)
            image.save(root / "palette.png", transparency=0)
            torch.testing.assert_close(self.load_image(root / "palette.png", 4)[3], torch.zeros(2, 3))
            Image.new("RGB", (3, 2), "red").save(root / "rgb.jpg")
            torch.testing.assert_close(self.load_image(root / "rgb.jpg", 4)[3], torch.ones(2, 3))

    def test_rgb_and_grayscale_match_existing_mgds_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            Image.new("RGBA", (3, 2), (51, 117, 219, 0)).save(path)
            for channels in (1, 3):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(channels=channels, dtype=dtype):
                        expected = self.load_image(path, channels, MGDSLoadImage, dtype)
                        actual = self.load_image(path, channels, dtype=dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(actual.shape[0], channels)

    def test_models_default_to_rgb(self):
        for model_type in ModelType:
            self.assertEqual(BaseModel(model_type).image_channels, 3)

    def augment(self, cls, image, enabled=True, fixed=False):
        inputs = {"image": image, "enabled": enabled, "fixed": fixed, "strength": 0.4}
        augmentation = cls(["image"], "enabled", "fixed", "strength")
        augmentation._get_previous_item = lambda _variation, name, _index: inputs[name]
        augmentation._get_rand = lambda *_: Random(42)
        return augmentation.get_item(0, 0)["image"]

    def test_color_augmentations_preserve_alpha_and_match_rgb_for_images_and_videos(self):
        generator = torch.Generator().manual_seed(3)
        for cls in (RandomBrightness, RandomContrast, RandomSaturation, RandomHue):
            legacy_cls = getattr(importlib.import_module(f"mgds.pipelineModules.{cls.__name__}"), cls.__name__)
            for frames in (None, 1, 2):
                shape = (4, 6, 7) if frames is None else (4, frames, 6, 7)
                image = torch.rand(shape, generator=generator)
                original = image.clone()
                for enabled, fixed in ((True, False), (False, True), (False, False)):
                    with self.subTest(augmentation=cls.__name__, frames=frames, enabled=enabled, fixed=fixed):
                        result = self.augment(cls, image, enabled, fixed)
                        rgb = image[:3].movedim(0, -3)
                        expected = self.augment(legacy_cls, rgb, enabled, fixed).movedim(-3, 0)
                        torch.testing.assert_close(result[:3], expected, rtol=0, atol=0)
                        torch.testing.assert_close(result[3], image[3], rtol=0, atol=0)
                        torch.testing.assert_close(image, original, rtol=0, atol=0)
                        torch.testing.assert_close(self.augment(cls, image[:3], enabled, fixed), expected, rtol=0, atol=0)

    def test_single_frame_hue_acts_on_rgb_channels(self):
        red = torch.tensor([1.0, 0.0, 0.0, 0.25]).view(4, 1, 1, 1).expand(4, 1, 2, 2)
        result = self.augment(RandomHue, red, enabled=False, fixed=True)
        expected = functional.adjust_hue(red[:3, 0], 0.2)
        torch.testing.assert_close(result[:3, 0], expected)
        self.assertFalse(torch.equal(result[:3], red[:3]))

    def test_png_jpeg_and_cloud_output(self):
        image = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
        image.putpixel((0, 0), (10, 20, 30, 85))
        output = ModelSamplerOutput(FileType.IMAGE, image)
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / "sample")
            BaseModelSampler.save_sampler_output(output, destination, ImageFormat.PNG, None, None)
            with Image.open(destination + ".png") as png:
                self.assertEqual(png.mode, "RGBA")
                self.assertEqual(png.tobytes(), image.tobytes())
            BaseModelSampler.save_sampler_output(output, destination, ImageFormat.JPG, None, None)
            with Image.open(destination + ".jpg") as jpg:
                self.assertEqual(jpg.mode, "RGB")
                self.assertEqual(jpg.getpixel((15, 15)), (255, 255, 255))
        self.assertEqual(output.data.mode, "RGBA")
        restored = pickle.loads(pickle.dumps(output))
        self.assertEqual(restored.data.mode, "RGBA")
        self.assertEqual(restored.data.tobytes(), image.tobytes())
        rgb = ModelSamplerOutput(FileType.IMAGE, image.convert("RGB"))
        with Image.open(io.BytesIO(rgb.__reduce__()[1][1])) as packed:
            self.assertEqual(packed.format, "JPEG")

    def test_concept_preview_retains_alpha_only_for_supported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGBA", (16, 16), (100, 120, 200, 85)).save(root / "rgba.png")
            Image.new("L", (16, 16), 0).save(root / "rgba-masklabel.png")
            concept = ConceptConfig.default_values()
            concept.path = directory
            concept.image.enable_fixed_hue = True
            concept.image.random_hue_max_strength = 0.25
            for model_type in (ModelType.ANIMA, ModelType.ANIMA_QWEN21_VAE):
                config = TrainConfig.default_values()
                config.model_type = model_type
                for augmentations in (False, True):
                    with self.subTest(model_type=model_type, augmentations=augmentations):
                        image, _, _ = ConceptWindowController(config, concept).get_preview_image(0, augmentations)
                        expected_mode = "RGBA" if model_type == ModelType.ANIMA_QWEN21_VAE else "RGB"
                        self.assertEqual(image.mode, expected_mode)
                        if expected_mode == "RGBA":
                            self.assertEqual(image.getchannel("A").getextrema(), (85, 85))


if __name__ == "__main__":
    unittest.main()
