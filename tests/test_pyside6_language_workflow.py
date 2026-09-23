import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from modules.ui.PySide6TrainUIView import PySide6TrainView
from modules.util.ui.pyside6_i18n import saved_language, set_language


class LanguageWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_sidebar_switch_translates_pages_and_preserves_model_choice(self):
        old_dir = Path.cwd()
        with TemporaryDirectory() as scratch:
            settings = QSettings(str(Path(scratch) / "language.ini"), QSettings.Format.IniFormat)
            with patch("modules.util.ui.pyside6_i18n.language_settings", return_value=settings):
                view = None
                try:
                    os.chdir(scratch)
                    for name in ("training_presets", "training_concepts", "training_samples"):
                        Path(name).mkdir()
                    set_language(self.app, "en", persist=False)
                    view = PySide6TrainView()
                    view.show()
                    self.app.processEvents()
                    original_model = view.top_bar_component._model_combo.currentData()

                    view.navigation.russian_button.click()
                    self.app.processEvents()
                    self.assertEqual(saved_language(settings), "ru")
                    self.assertEqual(view.navigation._items["general"].text(), "Обзор")
                    self.assertEqual(view.page_heading.text(), "Настройки обучения")
                    self.assertEqual(view.top_bar_component._model_combo.currentData(), original_model)
                    for button in (
                        view.top_bar_component._preset_button,
                        view.top_bar_component._load_button,
                        view.top_bar_component._save_button,
                        view.training_tab.optimizer_adv_comp,
                        view.training_tab.lr_scheduler_adv_comp,
                    ):
                        self.assertGreaterEqual(button.width(), button.sizeHint().width())

                    view.navigation.english_button.click()
                    self.app.processEvents()
                    self.assertEqual(saved_language(settings), "en")
                    self.assertEqual(view.navigation._items["general"].text(), "Overview")
                    self.assertEqual(view.page_heading.text(), "Training Configuration")
                    self.assertEqual(view.top_bar_component._model_combo.currentData(), original_model)
                finally:
                    if view is not None:
                        view.controller._stop_always_on_tensorboard()
                        view.close()
                    set_language(self.app, "en", persist=False)
                    os.chdir(old_dir)


if __name__ == "__main__":
    unittest.main()
