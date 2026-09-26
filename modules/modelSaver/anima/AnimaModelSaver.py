import json
import os.path
import shutil
from pathlib import Path

from modules.model.anima import custom_vae, pipeline_anima21
from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch

from safetensors.torch import save_file


class AnimaModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: AnimaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.tokenizer, pipeline.t5_tokenizer)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if model.model_type == ModelType.ANIMA_QWEN21_VAE:
            # Diffusers needs these modules beside model_index.json to load the saved pipeline elsewhere.
            shutil.copyfile(custom_vae.__file__, os.path.join(destination, "custom_vae.py"))
            shutil.copyfile(custom_vae.__file__, os.path.join(destination, "vae", "custom_vae.py"))
            shutil.copyfile(pipeline_anima21.__file__, os.path.join(destination, "pipeline_anima21.py"))
            index_path = os.path.join(destination, "model_index.json")
            with open(index_path, encoding="utf-8") as handle:
                index = json.load(handle)
            index["_class_name"] = ["pipeline_anima21", "Anima21Pipeline"]
            index["vae"] = ["custom_vae", "AutoencoderKLQwenImage21"]
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)
                handle.write("\n")

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: AnimaModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # convert the diffusers transformer keys back to the original Anima format (net.*)
        state_dict = convert(model.transformer.state_dict(), model.checkpoint_diffusers_to_original())
        # the original checkpoint bundles the text conditioner under net.llm_adapter.*; its keys are
        # identical to the diffusers module, so only a prefix is needed.
        for key, value in model.text_conditioner.state_dict().items():
            state_dict["net.llm_adapter." + key] = value

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: AnimaModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: AnimaModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
