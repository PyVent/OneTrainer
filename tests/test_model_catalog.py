import unittest

from modules.util.enum.ModelType import ModelType
from modules.util.model_catalog import MODEL_TYPE_CHOICES


class ModelCatalogTest(unittest.TestCase):
    def test_catalog_has_unique_labels_and_all_selectable_models(self):
        labels, models = zip(*MODEL_TYPE_CHOICES, strict=True)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(models), len(set(models)))
        legacy_config_variants = {
            ModelType.STABLE_DIFFUSION_20_BASE,
            ModelType.STABLE_DIFFUSION_20_DEPTH,
            ModelType.STABLE_DIFFUSION_21_BASE,
        }
        self.assertEqual(set(models), set(ModelType) - legacy_config_variants)

    def test_anima_variants_have_distinct_labels_and_config_values(self):
        choices = dict(MODEL_TYPE_CHOICES)
        self.assertEqual(choices["Anima"], ModelType.ANIMA)
        self.assertEqual(choices["Anima (qwen 2.1 vae)"], ModelType.ANIMA_QWEN21_VAE)
        self.assertNotEqual(ModelType.ANIMA.value, ModelType.ANIMA_QWEN21_VAE.value)


if __name__ == "__main__":
    unittest.main()
