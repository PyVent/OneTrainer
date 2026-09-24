import unittest
from types import SimpleNamespace

from modules.util.LayerOffloadConductor import LayerOffloadConductor, LayerOffloadStrategy

import torch
from torch import nn


class LayerOffloadCpuTest(unittest.TestCase):
    def test_strategy_keeps_current_layer_loaded_and_transitions_consistently(self):
        layers = [nn.Linear(2, 2) for _ in range(4)]
        strategy = LayerOffloadStrategy(layers, 0.5)

        for direction, targets in (
            ((True, True), strategy.forward_forward_loaded_layers),
            ((True, False), strategy.forward_backward_loaded_layers),
            ((False, False), strategy.backward_forward_loaded_layers),
        ):
            for layer_index, target in enumerate(targets):
                with self.subTest(direction=direction, layer_index=layer_index):
                    self.assertIn(layer_index, target)
                    initially_loaded = list(range(len(layers)))
                    offloaded = strategy.get_layers_to_offload(layer_index, *direction, initially_loaded)
                    remaining = [index for index in initially_loaded if index not in offloaded]
                    loaded = strategy.get_layers_to_load(layer_index, *direction, remaining)
                    self.assertEqual(sorted(remaining + loaded), target)

    def test_materialize_with_no_offloadable_layers_does_not_crash(self):
        model = nn.Linear(2, 2)
        config = SimpleNamespace(train_device="cpu", temp_device="cpu", async_offloading=False)
        part = SimpleNamespace(offload_fraction=0.5, activation_offloading=False)
        conductor = LayerOffloadConductor(model, config, part)

        conductor.materialize()
        with torch.no_grad():
            output = model(torch.ones(1, 2))
        conductor.evict()

        self.assertEqual(output.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
