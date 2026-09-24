import unittest

from modules.util.config.BaseConfig import BaseConfig, ConfigValidationError


class ConfigValidationTest(unittest.TestCase):
    def test_boolean_text_is_parsed_instead_of_becoming_truthy(self):
        config = BaseConfig([("enabled", True, bool, False)])

        config.from_dict({"enabled": "false"}, strict=True)
        self.assertIs(config.enabled, False)
        config.from_dict({"enabled": "yes"}, strict=True)
        self.assertIs(config.enabled, True)

        with self.assertRaisesRegex(ConfigValidationError, "enabled"):
            config.from_dict({"enabled": "flase"}, strict=True)

    def test_invalid_number_is_reported_instead_of_using_default(self):
        config = BaseConfig([("steps", 7, int, False)])

        with self.assertRaisesRegex(ConfigValidationError, "steps"):
            config.from_dict({"steps": "many"}, strict=True)
        self.assertEqual(config.steps, 7)

    def test_partial_update_and_nested_field_path(self):
        config = BaseConfig([
            ("epochs", 7, int, False),
            ("optimizer", BaseConfig([("momentum", 0.9, float, False)]), BaseConfig, False),
        ])

        config.from_dict({"optimizer": {"momentum": "0.8"}}, strict=True)
        self.assertEqual(config.optimizer.momentum, 0.8)
        self.assertEqual(config.epochs, 7)

        with self.assertRaisesRegex(ConfigValidationError, r"optimizer\.momentum"):
            config.from_dict({"optimizer": {"momentum": "bad"}}, strict=True)

    def test_nullable_boolean_remains_nullable(self):
        config = BaseConfig([("enabled", True, bool, True)])
        config.from_dict({"enabled": None}, strict=True)
        self.assertIsNone(config.enabled)

    def test_future_version_is_rejected(self):
        config = BaseConfig([], config_version=1)
        with self.assertRaisesRegex(ConfigValidationError, "version"):
            config.from_dict({"__version": config.config_version + 1}, strict=True)


if __name__ == "__main__":
    unittest.main()
