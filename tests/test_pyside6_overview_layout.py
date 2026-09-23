import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QGroupBox, QLabel, QLineEdit, QScrollArea

from modules.ui.PySide6TrainUIView import PySide6TrainView


class OverviewLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        previous_dir = Path.cwd()
        test_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(test_dir.cleanup)
        cls.addClassCleanup(os.chdir, previous_dir)
        os.chdir(test_dir.name)
        for name in ("training_presets", "training_concepts", "training_samples"):
            Path(name).mkdir()
        (Path("training_concepts") / "concepts.json").write_text("[]", encoding="utf-8")
        (Path("training_samples") / "samples.json").write_text("[]", encoding="utf-8")
        cls.view = PySide6TrainView()
        cls.view._show_navigation_page("general")
        cls.view.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.view.controller._stop_always_on_tensorboard()
        cls.view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_cards_keep_every_field_and_binding(self):
        page = self.view._tab_widgets["general"]
        cards = page.findChildren(QGroupBox)
        self.assertEqual(
            [card.title() for card in cards],
            ["Paths and run safety", "Monitoring and validation", "Devices and performance"],
        )
        self.assertTrue(all(card.findChild(QGridLayout, "overviewFieldGrid") for card in cards))
        self.assertTrue(all(card.findChild(QLabel, "pageSubtitle").text() for card in cards))

        before = {id(widget) for card in cards for cls in (QCheckBox, QLineEdit) for widget in card.findChildren(cls)}
        self.assertTrue(before)
        self.view.resize(1550, 800)
        self.app.processEvents()
        self.app.processEvents()
        self.view.resize(940, 800)
        self.app.processEvents()
        self.app.processEvents()
        after = {id(widget) for card in cards for cls in (QCheckBox, QLineEdit) for widget in card.findChildren(cls)}
        self.assertEqual(before, after)

        var = self.view.ui_state.get_var("prevent_overwrites")
        value = bool(var.get())
        checkbox = cards[0].findChild(QGridLayout, "overviewFieldGrid").itemAtPosition(4, 1).widget()
        self.assertIsInstance(checkbox, QCheckBox)
        # The field remains connected in both directions after the resize.
        checkbox.setChecked(not value)
        self.assertEqual(bool(var.get()), not value)
        var.set(value)
        self.assertEqual(checkbox.isChecked(), value)

    def test_fields_change_between_one_and_two_columns(self):
        page = self.view._tab_widgets["general"]
        scroll = page.findChild(QScrollArea)
        cards = page.findChildren(QGroupBox)
        self.view.resize(1550, 800)
        self.app.processEvents()
        self.app.processEvents()
        self.assertTrue(all(card.findChild(QGridLayout, "overviewFieldGrid").itemAtPosition(0, 2) for card in cards))
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)

        self.view.resize(940, 800)
        self.app.processEvents()
        self.app.processEvents()
        self.assertTrue(all(card.findChild(QGridLayout, "overviewFieldGrid").itemAtPosition(0, 2) is None for card in cards))
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
