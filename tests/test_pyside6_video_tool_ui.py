import os
import tempfile
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QLineEdit, QTabWidget, QTextEdit

from modules.ui.PySide6VideoToolUIView import PySide6VideoToolUIView
from modules.ui.VideoToolUIController import VideoToolUIController


class VideoToolUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_time_ranges_and_status_preview_bindings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            previous_dir = os.getcwd()
            os.chdir(temporary_dir)
            try:
                controller = VideoToolUIController()
                window = PySide6VideoToolUIView(None, controller)
                self.addCleanup(window.close)
                window.show()
                self.app.processEvents()

                tabs = window.findChild(QTabWidget)
                self.assertEqual(tabs.count(), 3)
                for tab_index, start_name in ((0, "clip_time_start"), (1, "image_time_start")):
                    tabs.setCurrentIndex(tab_index)
                    self.app.processEvents()
                    scroll = tabs.widget(tab_index)
                    frame = scroll.widget()
                    range_widget = frame.layout().itemAtPosition(1, 1).widget()
                    fields = range_widget.findChildren(QLineEdit)
                    self.assertEqual(len(fields), 2)
                    self.assertEqual([field.text() for field in fields], ["00:00:00", "99:99:99"])
                    self.assertGreaterEqual(fields[1].x(), fields[0].x() + fields[0].width())
                    fields[0].setText("01:00")
                    fields[0].editingFinished.emit()
                    self.assertEqual(controller.args[start_name], "01:00")

                download_args = tabs.widget(2).findChild(QTextEdit)
                download_args.setPlainText("--quiet")
                self.assertEqual(controller.args["download_args"], "--quiet")
                window.ui_state.get_var("download_args").set("--no-warnings")
                self.assertEqual(download_args.toPlainText(), "--no-warnings")

                worker = threading.Thread(target=lambda: (
                    window.update_status("Synthetic status"),
                    window.update_preview(Image.new("RGB", (12, 12), "red"), "Synthetic preview"),
                ))
                worker.start()
                worker.join()
                self.app.processEvents()
                self.assertIn("Synthetic status", window._status_box.toPlainText())
                self.assertEqual(window._preview_caption_label.text(), "Synthetic preview")
                self.assertFalse(window._preview_label.pixmap().isNull())
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
