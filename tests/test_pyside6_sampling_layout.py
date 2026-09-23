import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from modules.ui.PySide6SamplingTabView import PySide6SampleWidgetView
from modules.ui.PySide6TrainUIView import PySide6TrainView
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig


class SampleRowLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_row_reflows_without_losing_fields_or_callbacks(self):
        element = SampleConfig.default_values(TrainConfig.default_values().model_type)
        actions = []
        parent = QWidget()
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        row = PySide6SampleWidgetView(
            parent, element, 0,
            lambda i, state: actions.append(("open", i, state)),
            lambda i: actions.append(("remove", i)),
            lambda i: actions.append(("clone", i)),
            lambda: actions.append(("save",)),
        )
        layout.addWidget(row)
        self.addCleanup(parent.close)
        parent.show()

        fields = row._fields
        for width, expected_mode in ((1300, "wide"), (900, "medium"), (500, "compact"), (350, "compact")):
            with self.subTest(width=width):
                parent.resize(width, 300)
                self.app.processEvents()
                self.assertEqual(row._layout_mode, expected_mode)
                self.assertEqual(row._fields, fields)
                self.assertLessEqual(row.minimumSizeHint().width(), row.width())
                self.assertLessEqual(row.width(), parent.width())
        self.assertEqual(row.minimumWidth(), row.minimumSizeHint().width())

        row.width_entry.setText("640")
        self.assertIsNone(row.width_entry._validator.flush())
        row.width_entry.editingFinished.emit()
        self.assertEqual(element.width, 640)
        self.assertIn(("save",), actions)

        row.enabled_switch.click()
        self.assertFalse(element.enabled)
        self.assertFalse(row.width_entry.isEnabled())
        parent.resize(900, 300)
        self.app.processEvents()
        self.assertFalse(row.width_entry.isEnabled())
        row.enabled_switch.click()
        self.assertTrue(row.width_entry.isEnabled())
        self.assertEqual(row.width_entry.text(), "640")

        fields[0].click()
        fields[1].click()
        row.button.click()
        self.assertIn(("remove", 0), actions)
        self.assertIn(("clone", 0), actions)
        self.assertIn(("open", 0, row.ui_state), actions)

    def test_sampling_header_does_not_widen_all_pages(self):
        previous_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as scratch:
            try:
                os.chdir(scratch)
                for name in ("training_presets", "training_concepts", "training_samples"):
                    Path(name).mkdir()
                window = PySide6TrainView()
                page = window._tab_widgets["sampling"]
                header = page.layout().itemAtPosition(0, 0).widget()
                layout = header.layout()
                self.assertIsNotNone(layout.itemAtPosition(0, 3))
                self.assertIsNone(layout.itemAtPosition(0, 4))
                self.assertIsNotNone(layout.itemAtPosition(1, 3))
                self.assertIsNotNone(layout.itemAtPosition(2, 0))
                self.assertLess(page.minimumSizeHint().width(), 750)
                window.controller._stop_always_on_tensorboard()
                window.close()
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
