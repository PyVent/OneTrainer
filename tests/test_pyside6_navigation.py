import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from modules.util.ui.pyside6_navigation import WorkflowNavigation


class WorkflowNavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dynamic_pages_and_selection(self):
        navigation = WorkflowNavigation()
        selected = []
        navigation.page_selected.connect(selected.append)

        navigation.set_pages({"general", "model", "training", "LoRA", "sampling"})
        self.assertEqual(set(navigation._items), {"general", "model", "training", "LoRA", "sampling"})
        self.assertEqual(navigation.tree.topLevelItemCount(), 3)

        navigation.select_page("training")
        self.assertEqual(selected, [])  # synchronizing from QTabWidget must not loop back

        navigation.tree.setCurrentItem(navigation._items["LoRA"])
        self.assertEqual(selected, ["LoRA"])

        navigation.set_pages({"general", "model", "training", "embedding", "sampling"})
        self.assertNotIn("LoRA", navigation._items)
        self.assertIn("embedding", navigation._items)
        self.assertEqual(selected, ["LoRA"])

        navigation.tree.setCurrentItem(navigation._items["embedding"])
        self.assertEqual(selected, ["LoRA", "embedding"])
        self.assertEqual(navigation._items["embedding"].data(0, Qt.ItemDataRole.UserRole), "embedding")


if __name__ == "__main__":
    unittest.main()
