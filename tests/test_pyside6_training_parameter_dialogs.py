import os
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QScrollArea

from modules.ui.PySide6SchedulerParamsWindowView import PySide6SchedulerParamsWindowView
from modules.ui.PySide6TimestepDistributionWindowView import PySide6TimestepDistributionWindowView
from modules.ui.SchedulerParamsWindowController import SchedulerParamsWindowController
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui.PySide6UIState import PySide6UIState


class TrainingParameterDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_scheduler_has_one_scroll_area_and_retains_parameter_edits(self):
        config = TrainConfig.default_values()
        config.learning_rate_scheduler = LearningRateScheduler.CUSTOM
        config.scheduler_params = [{"key": "warmup", "value": "100"}]
        view = PySide6SchedulerParamsWindowView(
            None, SchedulerParamsWindowController(config), PySide6UIState(config)
        )
        self.addCleanup(view.close)
        view.resize(420, 400)
        view.show()
        self.app.processEvents()

        self.assertEqual(len(view.findChildren(QScrollArea)), 1)
        self.assertEqual(len(view._kv_params_view.widgets), 1)
        row = view._kv_params_view.widgets[0]
        self.assertEqual(row.key.text(), "warmup")
        row.value.setText("200")
        self.assertIsNone(row.value._validator.flush())
        row.value.editingFinished.emit()
        self.assertEqual(config.scheduler_params[0]["value"], "200")

        view.resize(360, 350)
        self.app.processEvents()
        self.assertEqual(row.value.text(), "200")
        self.assertLessEqual(view.minimumSizeHint().width(), view.width())

    def test_timestep_chart_moves_below_fields_and_keeps_bindings(self):
        config = TrainConfig.default_values()
        state = PySide6UIState(config)
        controller = TimestepDistributionWindowController(config)
        controller.generate_preview_data = Mock(return_value=list(range(100)))
        view = PySide6TimestepDistributionWindowView(None, controller, state)
        self.addCleanup(view.close)
        view.show()

        field_pairs = view._field_pairs
        min_strength = next(
            entry for entry in view.findChildren(QLineEdit)
            if entry._validator.var is state.get_var("min_noising_strength")
        )
        for width, layout_mode, fields_stacked in (
            (1100, "wide", False),
            (700, "compact", False),
            (440, "compact", True),
            (1100, "wide", False),
        ):
            with self.subTest(width=width):
                view.resize(width, 600)
                self.app.processEvents()
                self.assertEqual(view._layout_mode, layout_mode)
                self.assertEqual(view._fields_stacked, fields_stacked)
                self.assertEqual(view._field_pairs, field_pairs)
                self.assertEqual(view._scroll.horizontalScrollBar().maximum(), 0)

        min_strength.setText("0.2")
        self.assertIsNone(min_strength._validator.flush())
        self.assertEqual(config.min_noising_strength, 0.2)
        view.resize(440, 600)
        self.app.processEvents()
        self.assertEqual(min_strength.text(), "0.2")

        view._preview_frame.findChild(QPushButton).click()
        self.assertEqual(controller.generate_preview_data.call_count, 2)


if __name__ == "__main__":
    unittest.main()
