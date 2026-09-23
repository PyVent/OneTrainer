import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from modules.ui.PySide6TopBarView import PySide6TopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState


class _ControllerWithoutFiles(TopBarController):
    def load_preset_tree(self, dir="training_presets"):
        return []

    def load_config_from_file(self, filename):
        return None


class TopBarLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_layout_reflows_and_model_replaces_method_selector(self):
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
        self.assertIsNot(view.training_method, previous_method)
        self.assertIs(view._placed_method, view.training_method)


if __name__ == "__main__":
    unittest.main()
