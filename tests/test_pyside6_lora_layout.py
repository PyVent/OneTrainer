import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit

from modules.ui.LoraTabController import LoraTabController
from modules.ui.PySide6LoraTabView import PySide6LoraTabView
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState


class LoraLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_options_scroll_and_reflow_for_all_peft_types(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            previous_dir = os.getcwd()
            os.chdir(temporary_dir)
            try:
                config = TrainConfig.default_values()
                state = PySide6UIState(config)
                view = PySide6LoraTabView(None, LoraTabController(config), state)
                self.addCleanup(view.close)
                view.resize(700, 200)
                view.show()
                self.app.processEvents()

                self.assertEqual(view._option_columns, 1)
                self.assertGreater(view.scroll_area.verticalScrollBar().maximum(), 0)
                rank_field = next(
                    field for field in view.options_frame.findChildren(QLineEdit)
                    if getattr(field._validator, "var_name", None) == "lora_rank"
                )
                rank_field.setText("24")
                rank_field.editingFinished.emit()
                self.assertEqual(config.lora_rank, 24)

                for peft_type in ("LOHA", "OFT_2", "LOKR", "LORA"):
                    with self.subTest(peft_type=peft_type):
                        state.get_var("peft_type").set(peft_type)
                        self.app.processEvents()
                        self.assertGreater(len(view._option_cards), 0)
                        self.assertEqual(view._option_columns, 1)

                new_rank_field = next(
                    field for field in view.options_frame.findChildren(QLineEdit)
                    if getattr(field._validator, "var_name", None) == "lora_rank"
                )
                self.assertEqual(new_rank_field.text(), "24")
                view.resize(1200, 700)
                self.app.processEvents()
                self.assertEqual(view._option_columns, 2)
                self.assertEqual(view.scroll_area.horizontalScrollBar().maximum(), 0)
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
