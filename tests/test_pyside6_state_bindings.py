import unittest

from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.QtVar import QtVar

from PySide6.QtCore import QObject
from shiboken6 import delete


class StateBindingTest(unittest.TestCase):
    def test_owned_subscription_is_removed_with_qt_object(self):
        value = QtVar(0)
        owner = QObject()
        received = []
        value.subscribe(received.append, owner=owner)
        value.set(1)
        delete(owner)
        value.set(2)
        self.assertEqual(received, [1])

    def test_update_retargets_existing_top_level_and_nested_fields(self):
        original = TrainConfig.default_values()
        replacement = TrainConfig.default_values()
        state = PySide6UIState(original)
        state.update(replacement)

        original_rate = original.learning_rate
        original_adam_mode = original.optimizer.adam_w_mode
        state.get_var("learning_rate").set("0.00042")
        state.get_var("optimizer.adam_w_mode").set(not original_adam_mode)

        self.assertEqual(original.learning_rate, original_rate)
        self.assertEqual(original.optimizer.adam_w_mode, original_adam_mode)
        self.assertEqual(replacement.learning_rate, 0.00042)
        self.assertEqual(replacement.optimizer.adam_w_mode, not original_adam_mode)

    def test_optimizer_refresh_keeps_ui_and_saved_config_on_same_object(self):
        config = TrainConfig.default_values()
        state = PySide6UIState(config)
        training = TrainingTabController(config)
        optimizer_window = OptimizerParamsWindowController(config)

        for refresh in (
            training.restore_optimizer_config,
            optimizer_window.restore_optimizer_config,
            optimizer_window.load_defaults,
        ):
            refresh(state)
            self.assertIs(state.get_var("optimizer").obj, config.optimizer)
            state.get_var("optimizer.momentum").set("0.7")
            self.assertEqual(config.optimizer.momentum, 0.7)


if __name__ == "__main__":
    unittest.main()
