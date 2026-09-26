import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.PySide6TopBarView import PySide6TopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QApplication


class _ControllerWithoutFiles(TopBarController):
    def load_preset_tree(self, dir="training_presets"):
        return []

    def load_config_from_file(self, filename):
        return None


class _ControllerWithPreset(_ControllerWithoutFiles):
    def __init__(self, config):
        super().__init__(config)
        self.loaded_paths = []

    def load_preset_tree(self, dir="training_presets"):
        return [("Model", [("Built-in", "synthetic-preset.json")])]

    def load_config_from_file(self, filename):
        self.loaded_paths.append(filename)
        return


class TopBarLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_layout_reflows_and_model_updates_method_selector(self):
        config = TrainConfig.default_values()
        view = PySide6TopBarView(
            None,
            _ControllerWithoutFiles(config),
            PySide6UIState(config),
            lambda _model: None,
            lambda _method: None,
            lambda: None,
        )
        self.addCleanup(view.close)

        for width, mode in ((1300, "wide"), (1100, "medium"), (900, "medium"),
                            (800, "compact"), (540, "compact")):
            with self.subTest(width=width):
                view.resize(width, 180)
                view.show()
                self.app.processEvents()
                self.assertEqual(view._layout_mode, mode)
                self.assertEqual(view.width(), width)
                for widget in (view._model_combo, view.training_method,
                               view._preset_button, view._load_button, view._save_button):
                    self.assertLessEqual(widget.mapTo(view, widget.rect().bottomRight()).x(), width)
                self.assertTrue(view._model_combo.isVisible())
                self.assertTrue(view.training_method.isVisible())

        previous_method = view.training_method
        view._model_combo.setCurrentIndex(1)
        self.app.processEvents()
        self.assertIs(view.training_method, previous_method)
        self.assertEqual(
            [view.training_method.itemText(i) for i in range(view.training_method.count())],
            [label for label, _ in view.controller.get_training_methods(view.controller.train_config.model_type)],
        )
        self.assertEqual(
            view.training_method.itemData(view.training_method.currentIndex()),
            view.controller.train_config.training_method,
        )

    def test_nested_preset_action_loads_its_path(self):
        config = TrainConfig.default_values()
        controller = _ControllerWithPreset(config)
        view = PySide6TopBarView(
            None, controller, PySide6UIState(config),
            lambda _model: None, lambda _method: None, lambda: None,
        )
        self.addCleanup(view.close)
        controller.loaded_paths.clear()  # Ignore the optional last-session config.
        submenu = view._preset_button.menu().actions()[0].menu()
        with patch.object(view, "_show_load_error"):
            submenu.actions()[0].trigger()
        self.assertEqual(controller.loaded_paths, ["synthetic-preset.json"])


if __name__ == "__main__":
    unittest.main()
