import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QFormLayout, QLineEdit, QScrollArea

from modules.ui.CloudTabController import CloudTabController
from modules.ui.PySide6CloudTabView import PySide6CloudTabView
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState


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
        self.assertEqual([group.layout().rowCount() for group in groups], [10, 6, 6, 3, 6, 4])
        self.assertEqual(sum(group.layout().rowCount() for group in groups), 35)
        self.assertIsNotNone(self.view.gpu_types_menu)

        enabled = groups[0].layout().itemAt(0, QFormLayout.ItemRole.FieldRole).widget()
        self.assertIsInstance(enabled, QCheckBox)
        var = self.state.get_var("cloud.enabled")
        enabled.setChecked(not bool(var.get()))
        self.assertEqual(bool(var.get()), enabled.isChecked())

        remote_dir = groups[1].layout().itemAt(0, QFormLayout.ItemRole.FieldRole).widget()
        self.assertIsInstance(remote_dir, QLineEdit)
        remote_var = self.state.get_var("cloud.remote_dir")
        remote_var.set("/synthetic/remote")
        self.assertEqual(remote_dir.text(), "/synthetic/remote")

    def test_sections_reflow_without_horizontal_scroll(self):
        scroll = self.view.findChild(QScrollArea)
        self.view.resize(1400, 760)
        self.view.show()
        self.app.processEvents()
        self.app.processEvents()
        self.assertEqual(self.view._section_columns, 2)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)

        self.view.resize(760, 760)
        self.app.processEvents()
        self.app.processEvents()
        self.assertEqual(self.view._section_columns, 1)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertTrue(all(group.isVisible() for group in self.view._section_groups))


if __name__ == "__main__":
    unittest.main()
