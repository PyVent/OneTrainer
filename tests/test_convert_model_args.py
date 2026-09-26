import unittest

from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class ConvertModelArgsTest(unittest.TestCase):
    def setUp(self):
        self.args = ConvertModelArgs.default_values()
        self.args.input_name = "input.safetensors"
        self.args.output_model_destination = "output.safetensors"
        self.args.output_model_format = ModelFormat.SAFETENSORS

    def test_all_advertised_model_and_method_combinations_validate(self):
        for model_type in ModelType:
            self.args.model_type = model_type
            for method in self.args.supported_training_methods():
                self.args.training_method = method
                for output_format in model_type.supported_output_formats(method):
                    with self.subTest(model=model_type, method=method, output=output_format):
                        self.args.output_model_format = output_format
                        self.args.validate()

    def test_unsupported_embedding_and_vae_training_are_rejected(self):
        for model_type in (ModelType.ANIMA, ModelType.ANIMA_QWEN21_VAE, ModelType.QWEN, ModelType.FLUX_2):
            for method in (TrainingMethod.EMBEDDING, TrainingMethod.FINE_TUNE_VAE):
                with self.subTest(model=model_type, method=method):
                    self.args.model_type = model_type
                    self.args.training_method = method
                    with self.assertRaisesRegex(ValueError, "does not support"):
                        self.args.validate()

    def test_unsupported_output_and_missing_paths_are_rejected(self):
        self.args.model_type = ModelType.ANIMA
        with self.assertRaisesRegex(ValueError, "Unsupported output format"):
            self.args.validate()
        self.args.output_model_format = ModelFormat.DIFFUSERS
        self.args.input_name = " "
        with self.assertRaisesRegex(ValueError, "Input name"):
            self.args.validate()
        self.args.input_name = "model"
        self.args.output_model_destination = " "
        with self.assertRaisesRegex(ValueError, "output destination"):
            self.args.validate()

    def test_raw_embedding_conversion_and_internal_export_remain_supported(self):
        self.args.training_method = TrainingMethod.EMBEDDING
        self.args.base_model_name = ""
        self.args.validate()
        self.args.output_model_format = ModelFormat.INTERNAL
        self.args.validate()


if __name__ == "__main__":
    unittest.main()
