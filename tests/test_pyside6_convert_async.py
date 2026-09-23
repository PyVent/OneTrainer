import contextlib
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView


class _SyntheticController(ConvertModelUIController):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = True
        self.worker_thread_ids = []

    def perform_conversion(self):
        self.worker_thread_ids.append(threading.get_ident())
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("Synthetic worker was not released")
        if self.fail:
            raise ValueError("Synthetic conversion error")


class ConvertAsyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_controller_work_cleans_up_and_never_touches_view(self):
        controller = ConvertModelUIController()

        class View:
            def set_converting(self, _active):
                raise AssertionError("Worker touched a GUI view")

        controller.view = View()
        with patch("modules.ui.ConvertModelUIController.create.create_model_loader", side_effect=ValueError("synthetic load error")), \
                patch("modules.ui.ConvertModelUIController.torch_gc") as cleanup:
            with self.assertRaisesRegex(ValueError, "synthetic load error"):
                controller.perform_conversion()
        cleanup.assert_called_once_with()

    def test_convert_stays_responsive_and_recovers_after_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            previous_dir = os.getcwd()
            os.chdir(temporary_dir)
            controller = _SyntheticController()
            window = controller.create_window(None, PySide6ConvertModelUIView)
            main_thread_id = threading.get_ident()
            gui_updates = []
            original_set_converting = window.set_converting

            def record_gui_update(active):
                gui_updates.append(threading.get_ident())
                original_set_converting(active)

            window.set_converting = record_gui_update
            try:
                window.show()
                self.app.processEvents()
                with patch("modules.ui.PySide6ConvertModelUIView.traceback.print_exc"):
                    window.button.click()
                    self.assertTrue(controller.started.wait(1))
                    self.assertFalse(window.button.isEnabled())

                    marker = []
                    QTimer.singleShot(0, lambda: marker.append("GUI processed event"))
                    self.app.processEvents()
                    self.assertEqual(marker, ["GUI processed event"])

                    window.close()
                    self.assertTrue(window.isVisible())
                    controller.release.set()
                    self._wait_for(lambda: window._conversion_thread is None)
                    self.assertTrue(window.button.isEnabled())
                    self.assertIn("Synthetic conversion error", window._status_label.text())

                controller.started.clear()
                controller.release.clear()
                controller.fail = False
                window.button.click()
                self.assertTrue(controller.started.wait(1))
                self.assertFalse(window.button.isEnabled())
                controller.release.set()
                self._wait_for(lambda: window._conversion_thread is None)
                self.assertTrue(window.button.isEnabled())
                self.assertEqual(window._status_label.text(), "Model converted")
                self.assertEqual(gui_updates, [main_thread_id] * len(gui_updates))
                self.assertTrue(all(worker_id != main_thread_id for worker_id in controller.worker_thread_ids))
            finally:
                controller.release.set()
                if window._conversion_thread is not None:
                    self._wait_for(lambda: window._conversion_thread is None)
                with contextlib.suppress(RuntimeError):
                    window.close()
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
