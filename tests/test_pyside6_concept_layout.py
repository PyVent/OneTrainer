import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.PySide6ConceptTabView import PySide6ConceptTabView
from modules.ui.PySide6SamplingTabView import PySide6SamplingTabView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QApplication, QWidget


class ConceptLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        os.makedirs("training_concepts")
        os.makedirs("training_samples")

        concepts = []
        for index in range(12):
            concept = ConceptConfig.default_values()
            concept.name = f"Synthetic Item {index}"
            concept.type = ConceptType.VALIDATION if index % 2 else ConceptType.STANDARD
            concept.enabled = index % 3 != 0 and index != 11
            concepts.append(concept.to_dict())
        with open("training_concepts/concepts.json", "w", encoding="utf-8") as output:
            json.dump(concepts, output)

        sample = SampleConfig.default_values()
        sample.prompt = "Synthetic prompt"
        with open("training_samples/samples.json", "w", encoding="utf-8") as output:
            json.dump([sample.to_dict()], output)

    def tearDown(self):
        os.chdir(self.previous_dir)
        self.temp_dir.cleanup()

    def test_concepts_reflow_without_recreating_filtered_cards(self):
        config = TrainConfig.default_values()
        parent = QWidget()
        self.addCleanup(parent.close)
        view = PySide6ConceptTabView(parent, ConceptTabController(config), PySide6UIState(config))
        original_widgets = tuple(view.widgets)

        self.assertEqual(len(original_widgets), 12)
        self.assertLess(parent.minimumSizeHint().width(), 700)

        parent.resize(700, 600)
        parent.show()
        self.app.processEvents()
        narrow_columns = view._grid_columns
        self.assertGreaterEqual(narrow_columns, 2)

        view._search_entry.setText("Item 1")
        view._filter_combo.setCurrentText("VALIDATION")
        self.app.processEvents()
        self.assertEqual(view.filters["search"], "Item 1")
        self.assertEqual(view.filters["type"], "VALIDATION")
        self.assertEqual([w.i for w in view.widgets if not w.isHidden()], [1, 11])

        view._show_disabled_cb.setChecked(False)
        self.app.processEvents()
        self.assertFalse(view.filters["show_disabled"])
        self.assertEqual([w.i for w in view.widgets if not w.isHidden()], [1])

        parent.resize(1250, 700)
        self.app.processEvents()
        self.assertGreater(view._grid_columns, narrow_columns)
        self.assertEqual(tuple(view.widgets), original_widgets)
        self.assertEqual([w.i for w in view.widgets if not w.isHidden()], [1])
        self.assertEqual(view._search_entry.text(), "Item 1")
        self.assertEqual(view._filter_combo.currentText(), "VALIDATION")
        self.assertFalse(view._show_disabled_cb.isChecked())

    def test_sampling_still_uses_shared_config_list(self):
        config = TrainConfig.default_values()
        parent = QWidget()
        self.addCleanup(parent.close)
        view = PySide6SamplingTabView(parent, SamplingTabController(config), PySide6UIState(config))
        parent.show()
        self.app.processEvents()

        self.assertEqual(len(view.widgets), 1)
        self.assertEqual(view.widgets[0].prompt_entry.text(), "Synthetic prompt")
        self.assertIsNotNone(view.configs_dropdown)


if __name__ == "__main__":
    unittest.main()
