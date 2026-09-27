import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.modelSaver.anima.AnimaModelSaver import AnimaModelSaver
from modules.module.LoRAModule import DoRAModule, LoRAModule
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.merge_lora_util import merge_lora

import torch
from torch import nn

from safetensors.torch import save_file
from test_anima_qwen21_vae import MODEL_TYPES, small_model


class MergeLoraTest(unittest.TestCase):
    def test_merged_linear_and_convolution_match_live_lora_and_dora(self):
        for layer_kind in ("linear", "conv"):
            for adapter_kind in ("lora", "dora_input", "dora_output"):
                with self.subTest(layer=layer_kind, adapter=adapter_kind), torch.no_grad():
                    layer = nn.Linear(3, 4) if layer_kind == "linear" else nn.Conv2d(3, 4, 3, padding=1)
                    model = Mock()
                    model.model_type.denoising_model_part.return_value = "transformer"
                    model.transformer = nn.Sequential(layer)
                    model.lora_text_encoders.return_value = []
                    model.fusion_groups.return_value = None
                    adapter = LoRAModule("transformer.0", layer, 2, 1.0) if adapter_kind == "lora" else DoRAModule(
                        "transformer.0", layer, 2, 1.0, train_device=torch.device("cpu"), decompose_output_axis=adapter_kind == "dora_output",
                    )
                    adapter.lora_up.weight.normal_(0, 0.05)
                    adapter.hook_to_module()
                    x = torch.randn(2, 3) if layer_kind == "linear" else torch.randn(2, 3, 8, 8)
                    expected = layer(x)
                    model.lora_state_dict = adapter.state_dict(prefix="transformer.0.")
                    adapter.remove_hook_from_module()
                    bias = layer.bias.clone()
                    merge_lora(model)
                    torch.testing.assert_close(layer(x), expected)
                    torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)

    def test_fused_attention_update_is_split_into_the_base_projections(self):
        model = Mock()
        model.model_type.denoising_model_part.return_value = "transformer"
        model.transformer = nn.ModuleDict({"attn": nn.ModuleDict({key: nn.Linear(3, size) for key, size in (("q", 4), ("k", 2), ("v", 2))})})
        model.lora_text_encoders.return_value = []
        model.fusion_groups.return_value = [("attn", ["q", "k", "v"], "qkv", "qkv")]
        down, up = torch.randn(2, 3), torch.randn(8, 2)
        leaves = list(model.transformer["attn"].values())
        expected = torch.cat([layer.weight.detach() for layer in leaves]) + up @ down
        model.lora_state_dict = {"transformer.attn.qkv.lora_down.weight": down, "transformer.attn.qkv.lora_up.weight": up}
        merge_lora(model)
        torch.testing.assert_close(torch.cat([layer.weight.detach() for layer in leaves]), expected)

    def test_anima_merge_exports_complete_reloadable_model_for_both_vaes(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        for model_type in MODEL_TYPES:
            with self.subTest(model=model_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = small_model(model_type, root)
                base = root / "base-v1.0"
                AnimaModelSaver().save(model, ModelFormat.DIFFUSERS, str(base), torch.float32)
                name, layer = next((n, m) for n, m in model.transformer.named_modules() if isinstance(m, nn.Linear))
                down = torch.randn(2, layer.in_features) * 0.01
                up = torch.randn(layer.out_features, 2) * 0.01
                expected = layer.weight.detach().clone() + up @ down * 0.5
                lora = root / "adapter.safetensors"
                save_file({f"transformer.{name}.lora_down.weight": down, f"transformer.{name}.lora_up.weight": up,
                           f"transformer.{name}.alpha": torch.tensor(1.0)}, str(lora))
                controller = ConvertModelUIController(model_type, str(base))
                args = controller.convert_model_args
                args.training_method = TrainingMethod.LORA
                args.merge_lora = True
                args.input_name = str(lora)
                args.output_dtype = DataType.FLOAT_32
                args.output_model_format = ModelFormat.DIFFUSERS
                args.output_model_destination = str(root / "merged-v1.0")
                with patch("modules.ui.ConvertModelUIController.huggingface_util.configure_hub"):
                    controller.perform_conversion()
                index_name = "modular_model_index.json" if model_type == ModelType.ANIMA else "model_index.json"
                self.assertTrue((Path(args.output_model_destination) / index_name).is_file())
                from modules.util import create
                from modules.util.config.TrainConfig import QuantizationConfig
                from modules.util.ModelNames import ModelNames
                loaded = create.create_model_loader(model_type, TrainingMethod.FINE_TUNE).load(
                    model_type, ModelNames(base_model=args.output_model_destination), args.weight_dtypes(), QuantizationConfig.default_values(),
                )
                torch.testing.assert_close(loaded.transformer.get_submodule(name).weight, expected)
                self.assertEqual(loaded.vae.config.in_channels if "in_channels" in loaded.vae.config else loaded.image_channels, loaded.image_channels)

    def test_unknown_or_incomplete_weights_are_not_silently_ignored(self):
        model = Mock()
        model.model_type.denoising_model_part.return_value = "transformer"
        model.transformer = nn.Sequential(nn.Linear(3, 4))
        model.lora_text_encoders.return_value = []
        model.fusion_groups.return_value = None
        before = model.transformer[0].weight.detach().clone()
        for state in ({"transformer.0.lora_down.weight": torch.ones(2, 3)}, {"transformer.0.hada_w1_a": torch.ones(2, 3)}):
            model.lora_state_dict = state
            with self.assertRaises(ValueError):
                merge_lora(model)
            torch.testing.assert_close(model.transformer[0].weight, before)


if __name__ == "__main__":
    unittest.main()
