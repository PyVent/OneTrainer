import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modules.module.LoRAModule import LoRAModule, LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.riemannion_util import (
    build_riemannion_pair_map,
    create_riemannion_optimizer,
)


def make_model_and_groups():
    wrapper = object.__new__(LoRAModuleWrapper)
    wrapper.prefix = "unet"
    wrapper.lora_modules = {}
    groups = NamedParameterGroupCollection()

    for name, down, up, lr in (
        ("linear", nn.Linear(8, 2, bias=False), nn.Linear(2, 8, bias=False), 1e-3),
        ("conv", nn.Conv2d(4, 2, 3, bias=False), nn.Conv2d(2, 8, 1, bias=False), 2e-3),
    ):
        nn.init.zeros_(up.weight)
        layer = LoRAModule(name, None, 2, 2)
        layer.lora_down = down
        layer.lora_up = up
        wrapper.lora_modules[name] = layer
        extra = nn.Parameter(torch.ones(8))
        groups.add_group(NamedParameterGroup(name, [down.weight, up.weight, extra], lr))

    return SimpleNamespace(adapter=wrapper), groups


class RiemannionIntegrationTest(unittest.TestCase):
    def test_adapter_interface_and_rank_error(self):
        wrapper = object.__new__(LoRAModuleWrapper)
        wrapper.prefix = "sana"
        layer = LoRAModule("small", None, 3, 3)
        layer.lora_down = nn.Linear(4, 3, bias=False)
        layer.lora_up = nn.Linear(3, 4, bias=False)
        wrapper.lora_modules = {"small": layer}
        model = SimpleNamespace(container=[wrapper], adapters=lambda: [wrapper])
        with self.assertRaisesRegex(RuntimeError, "Reduce the LoRA rank"):
            build_riemannion_pair_map(model)

        layer.lora_down = nn.Linear(8, 3, bias=False)
        layer.lora_up = nn.Linear(3, 8, bias=False)
        self.assertEqual(len(build_riemannion_pair_map(model)), 2)

    def test_existing_config_keeps_optimizer_settings(self):
        config = TrainConfig.default_values()
        config.optimizer.from_dict({
            "optimizer": "RIEMANNION",
            "riemannion_init_scale": 1e-6,
            "riemannion_sigma_floor": 1e-8,
        })
        self.assertEqual(config.optimizer.optimizer, Optimizer.RIEMANNION)
        self.assertEqual(config.optimizer.riemannion_init_scale, 1e-6)
        self.assertEqual(config.optimizer.riemannion_sigma_floor, 1e-8)

    def test_training_step_and_resume(self):
        config = TrainConfig.default_values()
        config.optimizer.optimizer = Optimizer.RIEMANNION
        config.optimizer.momentum = 0.9
        config.optimizer.weight_decay = 0.00316
        config.optimizer.riemannion_init_scale = 1e-6
        config.optimizer.riemannion_sigma_floor = 1e-8

        model, groups = make_model_and_groups()
        pair_map = build_riemannion_pair_map(model)
        optimizer = create_riemannion_optimizer(
            groups.parameters_for_optimizer(config), pair_map, config
        )
        self.assertEqual([g["optim_type"] for g in optimizer.param_groups],
                         ["riemannion", "adam", "riemannion", "adam"])
        self.assertEqual([g["lr"] for g in optimizer.param_groups],
                         [1e-3, 1e-3, 2e-3, 2e-3])

        for param in groups.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()
        saved = optimizer.state_dict()
        resumed = create_riemannion_optimizer(
            groups.parameters_for_optimizer(config), pair_map, config
        )
        resumed.load_state_dict(saved)
        self.assertEqual(len(resumed.param_groups), 4)
        restored = resumed.state_dict()
        self.assertEqual(restored["state"].keys(), saved["state"].keys())
        for key, original_state in saved["state"].items():
            self.assertEqual(restored["state"][key]["step"], original_state["step"])
        before = [param.detach().clone() for param in groups.parameters()]
        for param in groups.parameters():
            param.grad = torch.ones_like(param)
        resumed.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in groups.parameters()))
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, groups.parameters())))


if __name__ == "__main__":
    unittest.main()
