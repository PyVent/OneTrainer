import copy
import unittest
from types import SimpleNamespace

from modules.util.checkpointing_util import create_checkpoint, enable_checkpointing

import torch
from torch import nn


class FakeConductor:
    def __init__(self):
        self.calls = []

    def add_layer(self, layer, indices):
        self.calls.append(("add", indices))

    def offload_activated(self):
        return True

    def start_forward(self, backward):
        self.calls.append(("start", backward))

    def before_layer(self, index, call_id, args):
        self.calls.append(("before", index))
        return args

    def after_layer(self, index, call_id, args):
        self.calls.append(("after", index))


class CheckpointingCpuTest(unittest.TestCase):
    def test_checkpointed_linear_matches_regular_forward_and_backward(self):
        torch.manual_seed(19)
        ordinary = nn.Linear(3, 2)
        checkpointed = copy.deepcopy(ordinary)
        wrapped = create_checkpoint(checkpointed, torch.device("cpu"))
        inputs = torch.randn(4, 3)

        expected = ordinary(inputs).square().sum()
        actual = wrapped(inputs).square().sum()
        expected.backward()
        actual.backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(checkpointed.weight.grad, ordinary.weight.grad)
        torch.testing.assert_close(checkpointed.bias.grad, ordinary.bias.grad)

    def test_offload_without_checkpointing_runs_under_no_grad_only(self):
        conductor = FakeConductor()
        layer = create_checkpoint(
            nn.Linear(2, 2), torch.device("cpu"), conductor=conductor, checkpointing=False,
        )

        with torch.no_grad():
            result = layer(torch.ones(1, 2))
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual([call[0] for call in conductor.calls], ["add", "start", "before", "after"])

        with self.assertRaisesRegex(NotImplementedError, "requires gradient checkpointing"):
            layer(torch.ones(1, 2))

    def test_offload_preserves_keyword_arguments_and_defaults(self):
        class ScaledLayer(nn.Module):
            def forward(self, value, scale=2):
                return value * scale

        layer = create_checkpoint(
            ScaledLayer(), torch.device("cpu"), conductor=FakeConductor(), checkpointing=False,
        )
        with torch.no_grad():
            torch.testing.assert_close(layer(torch.tensor([3.0])), torch.tensor([6.0]))
            torch.testing.assert_close(layer(torch.tensor([3.0]), scale=4), torch.tensor([12.0]))

    def test_empty_module_list_is_skipped(self):
        model = nn.Module()
        model.layers = nn.ModuleList()
        config = SimpleNamespace(train_device="cpu")
        part = SimpleNamespace(
            checkpointing_or_offloading_enabled=lambda: True,
            offloading_enabled=lambda: False,
            checkpointing_enabled=lambda: True,
        )

        conductor = enable_checkpointing(model, config, part, False, [(nn.Linear, [])])

        self.assertIsNone(conductor)
        self.assertEqual(len(model.layers), 0)

    def test_nonempty_module_list_is_checkpointed(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 1)])
        config = SimpleNamespace(train_device="cpu")
        part = SimpleNamespace(
            checkpointing_or_offloading_enabled=lambda: True,
            offloading_enabled=lambda: False,
            checkpointing_enabled=lambda: True,
        )

        enable_checkpointing(model, config, part, False, [(nn.Linear, [])])
        output = model.layers[1](model.layers[0](torch.ones(1, 2))).sum()
        output.backward()

        self.assertIsNotNone(model.layers[0].weight.grad)
        self.assertIsNotNone(model.layers[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
