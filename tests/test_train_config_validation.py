import unittest

from modules.util.config.BaseConfig import ConfigValidationError
from modules.util.config.TrainConfig import TrainConfig


class TrainConfigValidationTest(unittest.TestCase):
    def test_current_and_previous_version_configs_load(self):
        current = TrainConfig.default_values().to_dict()
        TrainConfig.default_values().from_dict(current, strict=True)

        previous = {
            "__version": 10,
            "model_type": "STABLE_DIFFUSION_15",
            "gradient_checkpointing": "ON",
            "unet": {},
        }
        restored = TrainConfig.default_values().from_dict(previous, strict=True)
        self.assertTrue(restored.unet.gradient_checkpointing)

    def test_future_version_is_rejected(self):
        config = TrainConfig.default_values()
        with self.assertRaisesRegex(ConfigValidationError, "version"):
            config.from_dict({"__version": config.config_version + 1}, strict=True)


if __name__ == "__main__":
    unittest.main()
