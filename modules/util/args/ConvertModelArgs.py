import argparse
import os
from typing import Any

from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class ConvertModelArgs(BaseArgs):
    model_type: ModelType
    training_method: TrainingMethod
    input_name: str
    base_model_name: str
    huggingface_token: str
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str
    merge_lora: bool

    def output_training_method(self) -> TrainingMethod:
        return TrainingMethod.FINE_TUNE if self.training_method == TrainingMethod.LORA and self.merge_lora else self.training_method

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes.from_single_dtype(self.output_dtype)

    def supported_training_methods(self) -> tuple[TrainingMethod, ...]:
        return tuple(
            method for method in self.model_type.supported_training_methods()
            if method in (TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING)
        )

    def validate(self):
        if self.training_method not in self.supported_training_methods():
            raise ValueError(f"{self.model_type} does not support {self.training_method} conversion")
        formats = self.model_type.supported_output_formats(self.output_training_method())
        if self.output_model_format not in [*formats, ModelFormat.INTERNAL]:
            raise ValueError(
                f"Unsupported output format {self.output_model_format} for {self.model_type} / {self.training_method}"
            )
        if not self.input_name.strip():
            raise ValueError("Input name is required for model conversion")
        if not self.output_model_destination.strip():
            raise ValueError("Model output destination is required for model conversion")
        if self.merge_lora and self.training_method == TrainingMethod.LORA and not self.base_model_name.strip():
            raise ValueError("A base model is required to merge LoRA")
        if self.output_model_format.is_single_file() and not self.output_model_destination.lower().endswith(self.output_model_format.file_extension()):
            raise ValueError(f"{self.output_model_format} exports a file. Use a {self.output_model_format.file_extension()} output path.")
        output = os.path.normcase(os.path.abspath(self.output_model_destination))
        if any(output == os.path.normcase(os.path.abspath(path)) for path in (self.input_name, self.base_model_name) if path):
            raise ValueError("Choose an output path different from the input and base model")

    def model_names(self) -> ModelNames:
        return ModelNames(
            base_model=self.input_name,
            lora=self.input_name,
            embedding=[self.input_name],
        )

    @staticmethod
    def parse_args() -> 'ConvertModelArgs':
        parser = argparse.ArgumentParser(description="One Trainer Converter Script.")

        # @formatter:off

        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--training-method", type=TrainingMethod, required=False, default=TrainingMethod.FINE_TUNE, dest="training_method", help="The training method", choices=list(TrainingMethod))
        parser.add_argument("--input-name", type=str, required=True, dest="input_name", help="The model to convert")
        parser.add_argument("--merge-lora", action="store_true", help="Merge the input LoRA into its base model and export the resulting model")
        parser.add_argument("--base-model-name", type=str, required=False, default="", dest="base_model_name", help="Filename, directory or Hugging Face repository of the base model, needed to convert a LoRA or embedding")
        parser.add_argument("--huggingface-token", type=str, required=False, default="", dest="huggingface_token", help="Hugging Face token, needed to download a gated base model")
        parser.add_argument("--output-dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="output_dtype", help="The data type to save the output model", choices=list(DataType))
        parser.add_argument("--output-model-format", type=ModelFormat, required=True, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")

        # @formatter:on

        args = ConvertModelArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values():
        data = []

        data.append(("model_type", ModelType.STABLE_DIFFUSION_15, ModelType, False))
        data.append(("training_method", TrainingMethod.FINE_TUNE, TrainingMethod, False))
        data.append(("input_name", "", str, False))
        data.append(("base_model_name", "", str, False))
        data.append(("huggingface_token", "", str, False))
        data.append(("output_dtype", DataType.FLOAT_16, DataType, False))
        data.append(("output_model_format", None, ModelFormat, True))
        data.append(("output_model_destination", "", str, False))
        data.append(("merge_lora", False, bool, False))

        return ConvertModelArgs(data)
