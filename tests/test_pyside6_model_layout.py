import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QLineEdit

from modules.ui.PySide6TrainUIView import PySide6TrainView


class ModelLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _visible_label(view, text):
        return next(
            label for label in view.model_tab.findChildren(QLabel)
            if label.text() == text and label.isVisible()
        )

    def test_model_fields_survive_reflow_and_model_change(self):
        previous_dir = Path.cwd()
        test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(test_dir.cleanup)
        self.addCleanup(os.chdir, previous_dir)
        os.chdir(test_dir.name)
        for name in ("training_presets", "training_concepts", "training_samples"):
            Path(name).mkdir()
        view = PySide6TrainView()
        self.addCleanup(view.close)
        view.tabview.setCurrentWidget(view.model_tab)
        view.resize(1400, 850)
        view.show()
        view.ui_state.get_var("model_type").set("STABLE_DIFFUSION_15")
        view.change_model_type(view.controller.train_config.model_type)
        self.app.processEvents()

        self.assertEqual(view.model_tab._column_count, 2)
        self._visible_label(view, "UNet Data Type")

        rank_field = next(
            field for field in view.model_tab.findChildren(QLineEdit)
            if field.isVisible() and getattr(field._validator, "var_name", None) == "quantization.svd_rank"
        )
        rank_field.setText("12")
        rank_field.editingFinished.emit()
        self.assertEqual(view.ui_state.get_var("quantization.svd_rank").get(), "12")

        view.model_tab.scroll_frame.resize(700, 700)
        self.app.processEvents()
        self.assertEqual(view.model_tab._column_count, 1)
        self.assertEqual(rank_field.text(), "12")

        view.ui_state.get_var("model_type").set("FLUX_DEV_1")
        view.change_model_type(view.controller.train_config.model_type)
        self.app.processEvents()
        self._visible_label(view, "Transformer Data Type")
        self.assertEqual(view.ui_state.get_var("quantization.svd_rank").get(), "12")

        new_rank_field = next(
            field for field in view.model_tab.findChildren(QLineEdit)
            if field.isVisible() and getattr(field._validator, "var_name", None) == "quantization.svd_rank"
        )
        self.assertEqual(new_rank_field.text(), "12")


if __name__ == "__main__":
    unittest.main()
