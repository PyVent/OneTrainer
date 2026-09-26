import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.CloudTabController import CloudTabController
from modules.ui.PySide6CloudTabView import PySide6CloudTabView
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QLineEdit, QScrollArea


class CloudLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        config = TrainConfig.default_values()
        self.state = PySide6UIState(config)
        self.view = PySide6CloudTabView(None, CloudTabController(config, None), self.state)
        self.addCleanup(self.view.deleteLater)

    def test_all_fields_and_actions_are_kept_in_sections(self):
        groups = self.view._section_groups
        self.assertEqual(
            [group.title() for group in groups],
            [
                "Connection",
                "Remote installation",
                "Create a cloud instance",
                "Training connection",
                "Download and cleanup",
                "Instance lifecycle",
            ],
        )
        self.assertEqual(
            [len(self.view._section_fields[group]) for group in groups],
            [10, 6, 6, 3, 6, 4],
        )
        self.assertEqual(sum(len(self.view._section_fields[group]) for group in groups), 35)
        self.assertTrue(all(isinstance(group.layout(), QGridLayout) for group in groups))
        self.assertIsNotNone(self.view.gpu_types_menu)
        self.assertEqual(self.view.reattach, self.view.controller.reattach)

        enabled = self.view._section_fields[groups[0]][0][1]
        self.assertIsInstance(enabled, QCheckBox)
        var = self.state.get_var("cloud.enabled")
        enabled.setChecked(not bool(var.get()))
        self.assertEqual(bool(var.get()), enabled.isChecked())

        remote_dir = self.view._section_fields[groups[1]][0][1]
        self.assertIsInstance(remote_dir, QLineEdit)
        remote_var = self.state.get_var("cloud.remote_dir")
        remote_var.set("/synthetic/remote")
        self.assertEqual(remote_dir.text(), "/synthetic/remote")

    def test_sections_reflow_without_horizontal_scroll(self):
        scroll = self.view.findChild(QScrollArea)
        self.view.show()
        # A 1440 px main window leaves roughly 1180 px for the Cloud page
        # after its navigation sidebar. That still creates two narrow groups.
        for width, columns, stacked in (
            (1440, 2, False),
            (1180, 2, True),
            (1040, 2, True),
            (760, 1, False),
            (520, 1, True),
        ):
            with self.subTest(width=width):
                self.view.resize(width, 760)
                self.app.processEvents()
                self.app.processEvents()
                self.assertEqual(self.view._section_columns, columns)
                self.assertEqual(self.view._fields_stacked, stacked)
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                self.assertTrue(all(group.isVisible() for group in self.view._section_groups))


if __name__ == "__main__":
    unittest.main()
