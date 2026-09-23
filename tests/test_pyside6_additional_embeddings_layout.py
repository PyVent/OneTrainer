import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QPushButton, QWidget

from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.PySide6AdditionalEmbeddingsTabView import PySide6AdditionalEmbeddingsTabView
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.ui.PySide6UIState import PySide6UIState


class AdditionalEmbeddingsLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fields_and_actions_survive_resizing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            previous_dir = os.getcwd()
            os.chdir(temporary_dir)
            try:
                config = TrainConfig.default_values()
                config.additional_embeddings = [
                    TrainEmbeddingConfig.default_values(),
                    TrainEmbeddingConfig.default_values(),
                ]
                parent = QWidget()
                view = PySide6AdditionalEmbeddingsTabView(
                    parent,
                    AdditionalEmbeddingsTabController(config),
                    PySide6UIState(config),
                )
                self.addCleanup(parent.close)
                parent.resize(650, 650)
                parent.show()
                self.app.processEvents()

                first, second = view.widgets
                self.assertEqual(first._column_count, 1)
                field = next(
                    entry for entry in first.findChildren(QLineEdit)
                    if getattr(entry._validator, "var_name", None) == "placeholder"
                )
                field.setText("<synthetic>")
                field.editingFinished.emit()
                self.assertEqual(config.additional_embeddings[0].placeholder, "<synthetic>")

                parent.resize(1150, 700)
                self.app.processEvents()
                self.assertEqual(first._column_count, 2)
                self.assertIs(view.widgets[0], first)
                self.assertIs(view.widgets[1], second)
                self.assertEqual(field.text(), "<synthetic>")

                train_label = next(label for label in first.findChildren(QLabel) if label.text() == "train:")
                train_box = train_label.parentWidget().findChild(QCheckBox)
                train_box.click()
                self.assertFalse(config.additional_embeddings[0].train)

                clone_button = next(
                    button for button in first.findChildren(QPushButton)
                    if button.accessibleName() == "Clone embedding"
                )
                clone_button.click()
                self.app.processEvents()
                self.assertEqual(len(view.widgets), 3)

                remove_button = next(
                    button for button in first.findChildren(QPushButton)
                    if button.accessibleName() == "Remove embedding"
                )
                remove_button.click()
                self.app.processEvents()
                self.assertEqual(len(view.widgets), 2)
                self.assertIs(view.widgets[0], second)
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
