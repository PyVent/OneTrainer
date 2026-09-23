import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QFormLayout, QGroupBox, QLineEdit, QPushButton

from modules.ui.PySide6TrainUIView import PySide6TrainView
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui.validation import _active_validators


class DataAndBackupLayoutTest(unittest.TestCase):
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
        cls.view = PySide6TrainView()

    @classmethod
    def tearDownClass(cls):
        cls.view.controller._stop_always_on_tensorboard()
        cls.view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_data_fields_are_grouped_and_remain_bound(self):
        page = self.view._tab_widgets["data"]
        groups = page.findChildren(QGroupBox)
        self.assertEqual([group.title() for group in groups], ["Image preparation", "Latent cache"])
        self.assertEqual([group.layout().rowCount() for group in groups], [1, 2])

        checkbox = groups[0].layout().itemAt(0, QFormLayout.ItemRole.FieldRole).widget()
        self.assertIsInstance(checkbox, QCheckBox)
        var = self.view.ui_state.get_var("aspect_ratio_bucketing")
        original = bool(var.get())
        try:
            checkbox.setChecked(not original)
            self.assertEqual(bool(var.get()), not original)
            var.set(original)
            self.assertEqual(checkbox.isChecked(), original)
        finally:
            var.set(original)

    def test_backup_fields_and_actions_are_preserved(self):
        page = self.view._tab_widgets["backup"]
        groups = page.findChildren(QGroupBox)
        self.assertEqual([group.title() for group in groups], ["Automatic backups", "Model saves"])
        self.assertEqual([group.layout().rowCount() for group in groups], [5, 4])
        self.assertEqual(
            [button.text() for group in groups for button in group.findChildren(QPushButton)],
            ["backup now", "save now"],
        )

        filename = groups[1].layout().itemAt(2, QFormLayout.ItemRole.FieldRole).widget()
        self.assertIsInstance(filename, QLineEdit)
        var = self.view.ui_state.get_var("save_filename_prefix")
        original = str(var.get())
        try:
            filename.setText("qt-layout-check")
            filename.editingFinished.emit()
            self.assertEqual(str(var.get()), "qt-layout-check")
        finally:
            var.set(original)

    def test_removed_training_method_pages_release_validators(self):
        self.view.change_training_method(TrainingMethod.LORA)
        old_page = self.view._tab_widgets["LoRA"]
        old_validators = {
            field._validator for field in old_page.findChildren(QLineEdit)
            if hasattr(field, "_validator")
        }
        self.assertTrue(old_validators)

        self.view.change_training_method(TrainingMethod.FINE_TUNE)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertNotIn("LoRA", self.view._tab_widgets)
        self.assertEqual(old_validators & _active_validators, set())

        self.view.change_training_method(TrainingMethod.EMBEDDING)
        old_page = self.view._tab_widgets["embedding"]
        old_validators = {
            field._validator for field in old_page.findChildren(QLineEdit)
            if hasattr(field, "_validator")
        }
        self.assertTrue(old_validators)
        self.view.change_training_method(TrainingMethod.FINE_TUNE)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertNotIn("embedding", self.view._tab_widgets)
        self.assertEqual(old_validators & _active_validators, set())


if __name__ == "__main__":
    unittest.main()
