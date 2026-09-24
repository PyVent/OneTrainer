import json
import tempfile
import unittest
from pathlib import Path

from modules.modelLoader.vae_source_validation import validate_anima_vae_source


def write_safetensors_header(path: Path, channels: int, architecture: str = ""):
    header = {
        "__metadata__": {"modelspec.architecture": architecture},
        "decoder.conv1.weight": {
            "dtype": "F32",
            "shape": [1152, channels, 1, 3, 3],
            "data_offsets": [0, 0],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


class AnimaVaeSourceTest(unittest.TestCase):
    def test_model_repository_and_directory_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            validate_anima_vae_source("")
            validate_anima_vae_source("circlestone-labs/Anima-Base-v1.0-Diffusers")
            validate_anima_vae_source(temporary_directory)

    def test_qwen_image_21_file_reports_latent_channel_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "qwen_image_2.1_vae_bf16.safetensors"
            write_safetensors_header(source, 64, "qwen_image_2.1_vae")

            with self.assertRaisesRegex(ValueError, "64.*16"):
                validate_anima_vae_source(str(source))

    def test_single_file_vae_reports_supported_source_format(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "compatible.safetensors"
            write_safetensors_header(source, 16)

            with self.assertRaisesRegex(ValueError, "Diffusers.*directory"):
                validate_anima_vae_source(str(source))


if __name__ == "__main__":
    unittest.main()
