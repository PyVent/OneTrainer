import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.PySide6TrainUIView import PySide6TrainView
from modules.util.ui.pyside6_theme import apply_theme, saved_theme

from PySide6.QtCore import QSettings
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


class ThemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dark_and_light_palette_and_local_preference(self):
        with tempfile.TemporaryDirectory() as scratch:
            settings = QSettings(str(Path(scratch) / "theme.ini"), QSettings.Format.IniFormat)
            self.assertEqual(saved_theme(settings), "light")
            try:
                apply_theme(self.app, "dark", persist=True, settings=settings)
                self.assertEqual(saved_theme(settings), "dark")
                self.assertEqual(self.app.property("onetrainerTheme"), "dark")
                dark_window = self.app.palette().color(QPalette.ColorRole.Window)

                apply_theme(self.app, "light", persist=True, settings=settings)
                self.assertEqual(saved_theme(settings), "light")
                self.assertNotEqual(self.app.palette().color(QPalette.ColorRole.Window), dark_window)
            finally:
                apply_theme(self.app, "light")

    def test_sidebar_switch_updates_application_and_preference(self):
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as scratch:
            settings = QSettings(str(Path(scratch) / "theme.ini"), QSettings.Format.IniFormat)
            with patch("modules.util.ui.pyside6_theme.theme_settings", return_value=settings):
                try:
                    os.chdir(scratch)
                    for name in ("training_presets", "training_concepts", "training_samples"):
                        Path(name).mkdir()
                    view = PySide6TrainView()
                    view.navigation.dark_button.click()
                    self.assertEqual(self.app.property("onetrainerTheme"), "dark")
                    self.assertEqual(saved_theme(settings), "dark")
                    self.assertTrue(view.navigation.dark_button.isChecked())
                    view.controller._stop_always_on_tensorboard()
                    view.close()
                finally:
                    apply_theme(self.app, "light")
                    os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
