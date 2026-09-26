import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.util.ui import pyside6_components as ui
from modules.util.ui.pyside6_i18n import current_language, retranslate_tree, set_language
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QApplication, QWidget


class LocalizedComponentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous_language = current_language()
        set_language(self.app, "ru", persist=False)

    def tearDown(self):
        set_language(self.app, self.previous_language, persist=False)

    def test_key_value_combo_keeps_config_value_across_language_switch(self):
        config = {"mode": "light"}
        state = PySide6UIState(config)
        frame = QWidget()
        combo = ui.options_kv(
            frame, 0, 0, [("Light", "light"), ("Dark", "dark")], state, "mode"
        )
        self.assertEqual(combo.itemText(0), "Светлая")
        combo.setCurrentIndex(1)
        self.assertEqual(config["mode"], "dark")

        set_language(self.app, "en", persist=False)
        retranslate_tree(frame)
        self.assertEqual(combo.itemText(1), "Dark")
        self.assertEqual(combo.currentData(), "dark")
        self.assertEqual(config["mode"], "dark")

    def test_plain_combo_keeps_source_value(self):
        config = {"mode": "Light"}
        state = PySide6UIState(config)
        frame = QWidget()
        combo = ui.options(frame, 0, 0, ["Light", "Dark"], state, "mode")
        self.assertEqual(combo.currentText(), "Светлая")
        combo.setCurrentIndex(1)
        self.assertEqual(config["mode"], "Dark")


if __name__ == "__main__":
    unittest.main()
