import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit

from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState


class TrainingLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_columns_reflow_without_recreating_or_unbinding_fields(self):
        config = TrainConfig.default_values()
        state = PySide6UIState(config)
        view = PySide6TrainingTabView(None, TrainingTabController(config), state)
        self.addCleanup(view.close)
        view.show()

        learning_rate = next(
            entry for entry in view.findChildren(QLineEdit)
            if entry._validator.var is state.get_var("learning_rate")
        )
        field_count = len(view.findChildren(QLineEdit))
        columns = view._columns

        for width, expected_count in ((1600, 3), (1000, 2), (650, 1), (1600, 3)):
            with self.subTest(width=width):
                view.resize(width, 700)
                self.app.processEvents()
                self.assertEqual(view._column_count, expected_count)
                self.assertEqual(view._column_layout.count(), 3)
                self.assertEqual(view._columns, columns)
                self.assertEqual(len(view.findChildren(QLineEdit)), field_count)
                for index, column in enumerate(columns):
                    layout_index = view._column_layout.indexOf(column)
                    row, col, _, _ = view._column_layout.getItemPosition(layout_index)
                    self.assertEqual((row, col), divmod(index, expected_count))

        learning_rate.setText("0.0002")
        self.assertIsNone(learning_rate._validator.flush())
        self.assertEqual(state.get_var("learning_rate").get(), "0.0002")
        self.assertEqual(config.learning_rate, 0.0002)

        view.resize(650, 700)
        self.app.processEvents()
        self.assertEqual(learning_rate.text(), "0.0002")
        self.assertIs(learning_rate._validator.var, state.get_var("learning_rate"))


if __name__ == "__main__":
    unittest.main()
