import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.util.ui.pyside6_navigation import WorkflowNavigation

from PySide6.QtWidgets import QApplication


class WorkflowNavigationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dynamic_pages_and_selection(self):
        navigation = WorkflowNavigation()
        self.addCleanup(navigation.close)
        selected = []
        navigation.page_selected.connect(selected.append)

        navigation.set_pages({"general", "model", "training", "LoRA", "sampling"})
        self.assertEqual(set(navigation._items), {"general", "model", "training", "LoRA", "sampling"})
        self.assertFalse(navigation._items["training"].icon().isNull())

        navigation.select_page("training")
        self.assertEqual(selected, [])  # synchronizing from QTabWidget must not loop back
        self.assertTrue(navigation._items["training"].isChecked())

        navigation._items["LoRA"].click()
        self.assertEqual(selected, ["LoRA"])
        self.assertTrue(navigation._items["LoRA"].isChecked())

        navigation.set_pages({"general", "model", "training", "embedding", "sampling"})
        self.assertNotIn("LoRA", navigation._items)
        self.assertIn("embedding", navigation._items)
        self.assertEqual(selected, ["LoRA"])

        navigation._items["embedding"].click()
        self.assertEqual(selected, ["LoRA", "embedding"])

    def test_appearance_switch_and_help_are_real_actions(self):
        navigation = WorkflowNavigation()
        self.addCleanup(navigation.close)
        themes = []
        help_clicks = []
        navigation.theme_requested.connect(themes.append)
        navigation.help_requested.connect(lambda: help_clicks.append(True))

        navigation.set_theme("dark")
        self.assertTrue(navigation.dark_button.isChecked())
        navigation.light_button.click()
        self.assertEqual(themes, ["light"])
        navigation.help_button.click()
        self.assertEqual(help_clicks, [True])


if __name__ == "__main__":
    unittest.main()
