import contextlib
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QScrollArea

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.MuonAdamWindowController import MuonAdamWindowController
from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.ui.PySide6MuonAdamWindowView import PySide6MuonAdamWindowView
from modules.ui.PySide6OptimizerParamsWindowView import PySide6OptimizerParamsWindowView
from modules.ui.PySide6ProfilingWindowView import PySide6ProfilingWindowView
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui.PySide6UIState import PySide6UIState


class NestedToolWindowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.windows = []

    def tearDown(self):
        for window in reversed(self.windows):
            with contextlib.suppress(RuntimeError):
                window.close()
        os.chdir(self.previous_dir)
        self.temp_dir.cleanup()

    def test_convert_dynamic_fields_and_small_window_scroll(self):
        controller = ConvertModelUIController()
        window = PySide6ConvertModelUIView(None, controller)
        self.windows.append(window)
        window.resize(600, 200)
        window.show()
        self.app.processEvents()

        self.assertGreater(window._scroll_area.verticalScrollBar().maximum(), 0)
        self.assertEqual(window._layout.itemAtPosition(1, 0).widget().text(), "Training Method")
        method_combo = window._layout.itemAtPosition(1, 1).widget()
        self.assertIsInstance(method_combo, QComboBox)
        method_combo.setCurrentText("LoRA")
        self.app.processEvents()
        self.assertEqual(controller.convert_model_args.training_method, TrainingMethod.LORA)
        self.assertIn("Base Model Name", [label.text() for label in window._dynamic_frame.findChildren(QLabel)])

        method_combo.setCurrentText("Base Model")
        self.app.processEvents()
        self.assertEqual(controller.convert_model_args.training_method, TrainingMethod.FINE_TUNE)
        self.assertNotIn("Base Model Name", [label.text() for label in window._dynamic_frame.findChildren(QLabel)])

    def test_optimizer_and_muon_settings_scroll_and_toggle(self):
        config = TrainConfig.default_values()
        controller = OptimizerParamsWindowController(config)
        window = PySide6OptimizerParamsWindowView(None, controller, PySide6UIState(config))
        self.windows.append(window)
        window.resize(700, 220)
        window.show()
        self.app.processEvents()
        self.assertGreater(window.findChild(QScrollArea).verticalScrollBar().maximum(), 0)

        optimizer_combo = window._frame.layout().itemAtPosition(0, 1).widget()
        optimizer_combo.setCurrentText("MUON")
        self.app.processEvents()
        self.assertEqual(config.optimizer.optimizer, Optimizer.MUON)
        muon_label = next(
            label for label in window._dynamic_frame.findChildren(QLabel)
            if label.text() == "MuonWithAuxAdam"
        )
        label_layout = muon_label.parentWidget().layout()
        index = label_layout.indexOf(muon_label)
        row, column, _, _ = label_layout.getItemPosition(index)
        inline = label_layout.itemAtPosition(row, column + 1).widget()
        muon_switch = inline.findChild(QCheckBox)
        muon_switch.click()
        self.assertEqual(window.muon_adam_button.isEnabled(), muon_switch.isChecked())
        muon_switch.click()
        self.assertTrue(window.muon_adam_button.isEnabled())

        adam_config = TrainOptimizerConfig.default_values()
        adam_window = PySide6MuonAdamWindowView(
            None,
            MuonAdamWindowController(config, Optimizer.ADAMW_ADV),
            PySide6UIState(adam_config),
        )
        self.windows.append(adam_window)
        adam_window.resize(700, 220)
        adam_window.show()
        self.app.processEvents()
        self.assertGreater(adam_window.findChild(QScrollArea).verticalScrollBar().maximum(), 0)

    def test_profiling_controls_with_synthetic_controller(self):
        class Controller:
            def __init__(self):
                self.view = None
                self.dumped = False

            def dump_stack(self):
                self.dumped = True
                self.view.set_message("Synthetic stack")

            def start_profiler(self):
                self.view.set_profiling_active(True)

            def end_profiler(self):
                self.view.set_profiling_active(False)

        controller = Controller()
        window = PySide6ProfilingWindowView(None, controller)
        controller.view = window
        self.windows.append(window)
        window.show()
        self.app.processEvents()

        profile_button = window._profile_button
        profile_button.click()
        self.assertEqual(profile_button.text(), "End Profiling")
        profile_button.click()
        self.assertEqual(profile_button.text(), "Start Profiling")
        dump_button = next(button for button in window.findChildren(QPushButton) if button.text() == "Dump stack")
        dump_button.click()
        self.assertTrue(controller.dumped)
        self.assertEqual(window._message_label.text(), "Synthetic stack")


if __name__ == "__main__":
    unittest.main()
