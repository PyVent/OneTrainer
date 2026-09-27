import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.PySide6SampleWindowView import PySide6SampleWindowView
from modules.ui.SampleWindowController import SampleWindowController
from modules.util.config.TrainConfig import TrainConfig

from PIL import Image
from PySide6.QtWidgets import QApplication


class SamplingToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_standalone_generation_does_not_inherit_compile(self):
        config = TrainConfig.default_values()
        config.compile = True
        controller = SampleWindowController(config, False)
        self.assertFalse(controller.initial_train_config.compile)
        self.assertTrue(config.compile)

    def test_large_result_does_not_set_the_window_minimum_size(self):
        controller = SampleWindowController(TrainConfig.default_values(), False)
        window = PySide6SampleWindowView(None, controller)
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        before = window.minimumSizeHint()
        window._do_update_preview(Image.new("RGBA", (4096, 4096)))
        self.app.processEvents()
        self.assertLessEqual(window.minimumSizeHint().width(), before.width())
        self.assertLessEqual(window.minimumSizeHint().height(), before.height())
        window.resize(950, 700)
        self.app.processEvents()
        self.assertLess(window.width(), 1500)
        self.assertLessEqual(window._image_label.pixmap().width(), window._image_label.width())


if __name__ == "__main__":
    unittest.main()
