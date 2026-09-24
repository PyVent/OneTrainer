import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.modelLoader.AnimaModelLoader import AnimaModelLoader


class AnimaLoaderTest(unittest.TestCase):
    def test_incompatible_single_file_vae_fails_before_model_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "vae.safetensors"
            header = {
                "decoder.conv1.weight": {
                    "dtype": "F32",
                    "shape": [1152, 64, 1, 3, 3],
                    "data_offsets": [0, 0],
                },
            }
            encoded = json.dumps(header).encode("utf-8")
            source.write_bytes(len(encoded).to_bytes(8, "little") + encoded)
            model_names = SimpleNamespace(vae_model=str(source))

            with self.assertRaisesRegex(ValueError, "64.*16"):
                AnimaModelLoader().load(None, None, model_names, None, None)


if __name__ == "__main__":
    unittest.main()
