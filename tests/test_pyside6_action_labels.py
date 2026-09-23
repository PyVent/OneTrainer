import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.PySide6AdditionalEmbeddingsTabView import PySide6EmbeddingWidgetView
from modules.ui.PySide6ConceptTabView import PySide6ConceptWidgetView
from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView
from modules.ui.PySide6SamplingTabView import PySide6SampleWidgetView
from modules.ui.PySide6SchedulerParamsWindowView import PySide6KvWidget
from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.ui.PySide6UIState import PySide6UIState


class ActionLabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.previous_dir = Path.cwd()
        os.chdir(self.temporary_dir.name)

    def tearDown(self):
        os.chdir(self.previous_dir)
        self.temporary_dir.cleanup()

    def test_training_advanced_actions_have_readable_labels(self):
        config = TrainConfig.default_values()
        view = PySide6TrainingTabView(None, TrainingTabController(config), PySide6UIState(config))
        self.addCleanup(view.close)
        view.resize(650, 600)
        view.show()
        self.app.processEvents()

        for text in ("Optimizer", "Learning Rate Scheduler", "Timestep Distribution"):
            with self.subTest(text=text):
                label = next(label for label in view.findChildren(QLabel) if label.text() == text)
                grid = label.parentWidget().layout()
                row, column, _, _ = grid.getItemPosition(grid.indexOf(label))
                button = grid.itemAtPosition(row, column + 1).widget().findChild(QPushButton)
                self.assertEqual(button.text(), "Settings")
                self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 12)
        self.assertLess(view.minimumSizeHint().width(), 700)

    def test_repeated_item_actions_are_named_and_still_connected(self):
        config = TrainConfig.default_values()
        actions = []
        parent = QWidget()
        self.addCleanup(parent.close)

        sample = SampleConfig.default_values(config.model_type)
        sample_row = PySide6SampleWidgetView(
            parent, sample, 0,
            lambda i, state: actions.append("sample-edit"),
            lambda i: actions.append("sample-remove"),
            lambda i: actions.append("sample-copy"),
            lambda: None,
        )
        self.assertEqual([button.text() for button in (sample_row._fields[0], sample_row._fields[1], sample_row.button)],
                         ["Remove", "Copy", "Edit"])
        sample_row._fields[0].click()
        sample_row._fields[1].click()
        sample_row.button.click()

        concept = ConceptConfig.default_values()
        concept_row = PySide6ConceptWidgetView(
            parent, concept, 0,
            lambda *args: None,
            lambda i: actions.append("concept-remove"),
            lambda *args: actions.append("concept-copy"),
            lambda: None,
            ConceptTabController(config),
        )
        concept_buttons = {button.text(): button for button in concept_row.findChildren(QPushButton)}
        self.assertTrue({"Remove", "Copy"} <= concept_buttons.keys())
        parent.show()
        concept_row.show()
        self.app.processEvents()
        for button in concept_buttons.values():
            self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 10)
            self.assertGreaterEqual(button.height(), 30)
            self.assertLessEqual(button.geometry().right(), concept_row.width())
            button.click()

        embedding = TrainEmbeddingConfig.default_values()
        embedding_row = PySide6EmbeddingWidgetView(
            parent, embedding, 0,
            lambda *args: None,
            lambda i: actions.append("embedding-remove"),
            lambda *args: actions.append("embedding-copy"),
            lambda: None,
            Mock(randomize_uuid=lambda *args: None),
        )
        embedding_buttons = {button.text(): button for button in embedding_row.findChildren(QPushButton)}
        self.assertTrue({"Remove", "Copy"} <= embedding_buttons.keys())
        embedding_buttons["Remove"].click()
        embedding_buttons["Copy"].click()

        kv_row = PySide6KvWidget(
            parent, {"key": "synthetic", "value": "1"}, 0,
            lambda *args: None,
            lambda i: actions.append("parameter-remove"),
            lambda *args: None,
            lambda: None,
        )
        remove = next(button for button in kv_row.findChildren(QPushButton) if button.text() == "Remove")
        remove.click()
        self.assertEqual(actions, [
            "sample-remove", "sample-copy", "sample-edit",
            "concept-remove", "concept-copy",
            "embedding-remove", "embedding-copy", "parameter-remove",
        ])

    def test_batch_folder_buttons_explain_the_action(self):
        for view_class in (PySide6GenerateCaptionsWindowView, PySide6GenerateMasksWindowView):
            with self.subTest(view=view_class.__name__):
                view = view_class(None, Mock(), self.temporary_dir.name, False)
                self.addCleanup(view.close)
                button = next(button for button in view.findChildren(QPushButton) if button.text() == "Browse")
                self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance("Browse") + 12)


if __name__ == "__main__":
    unittest.main()
