import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


class CaptionUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name)

    def wait_for_dialog(self, view):
        deadline = time.monotonic() + 3
        while view._running and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertFalse(view._running, "Batch generation did not finish")

    def test_open_edit_save_and_close(self):
        Image.new("RGB", (40, 20), "red").save(self.path / "first.png")
        Image.new("RGB", (20, 40), "blue").save(self.path / "second.png")
        (self.path / "first.txt").write_text("original", encoding="utf-8")
        controller = CaptionUIController(str(self.path), False)
        controller.image_size = 100
        release_models = Mock()
        controller._release_models = release_models
        view = PySide6CaptionUIView(None, controller)
        view.show()
        self.app.processEvents()

        self.assertEqual(view.file_list.count(), 2)
        self.assertEqual(view.prompt_component.text(), "original")
        self.assertEqual(view.image_label.pixmap().size().width(), 100)

        view.enable_mask_editing.setChecked(True)
        QTest.mouseClick(view.image_label, Qt.MouseButton.LeftButton, pos=QPoint(30, 20))
        self.assertIsNotNone(controller.pil_mask)
        view.prompt_component.setText("new caption")
        view.save()
        self.assertEqual((self.path / "first.txt").read_text(encoding="utf-8"), "new caption")
        self.assertTrue((self.path / "first-masklabel.png").is_file())

        view.file_list.setCurrentRow(1)
        self.assertEqual(controller.current_image_index, 1)
        self.assertEqual(view.prompt_component.text(), "")
        view.brush_size.setValue(4)
        self.assertEqual(controller.mask_draw_radius, 0.04)
        view.toggle_mask()
        self.assertTrue(controller.display_only_mask)

        view.close()
        release_models.assert_called_once()

    def test_empty_folder(self):
        controller = CaptionUIController(str(self.path), False)
        controller._release_models = Mock()
        view = PySide6CaptionUIView(None, controller)
        self.assertEqual(view.file_list.count(), 0)
        self.assertEqual(view.image_label.text(), "No images in this folder")
        view.close()

    def test_batch_dialogs_forward_options(self):
        mask_controller = Mock()
        mask_view = PySide6GenerateMasksWindowView(None, mask_controller, str(self.path), True)
        mask_view.model.setCurrentText("Hex Color")
        mask_view.prompt.setText("subject")
        mask_view.create_masks()
        self.wait_for_dialog(mask_view)
        mask_controller.create_masks.assert_called_once_with(
            model_name="Hex Color", path=str(self.path), prompt="subject",
            mode_str="Create if absent", alpha_str="1.0", threshold_str="0.3",
            smooth_str="5", expand_str="10", include_subdirectories=True,
        )
        mask_view.set_progress(2, 4)
        self.assertEqual(mask_view.progress.value(), 2)
        mask_view.close()

        caption_controller = Mock()
        caption_view = PySide6GenerateCaptionsWindowView(None, caption_controller, str(self.path), False)
        caption_view.caption_prefix.setText("prefix")
        caption_view.create_captions()
        self.wait_for_dialog(caption_view)
        caption_controller.create_captions.assert_called_once_with(
            model_name="Blip", path=str(self.path), initial_caption="",
            caption_prefix="prefix", caption_postfix="",
            mode_str="Create if absent", include_subdirectories=False,
        )
        caption_view.close()

    def test_batch_dialogs_keep_ui_responsive_and_block_close(self):
        for view_cls, method_name, start_name in (
            (PySide6GenerateMasksWindowView, "create_masks", "create_masks"),
            (PySide6GenerateCaptionsWindowView, "create_captions", "create_captions"),
        ):
            with self.subTest(view=view_cls.__name__):
                started = threading.Event()
                release = threading.Event()
                controller = Mock()
                view = view_cls(None, controller, str(self.path), False)
                view.show()

                def generate(started=started, release=release, view=view, **_options):
                    started.set()
                    release.wait(timeout=2)
                    view.set_progress(2, 4)

                getattr(controller, method_name).side_effect = generate
                getattr(view, start_name)()
                try:
                    self.assertTrue(started.wait(timeout=1))
                    self.app.processEvents()
                    self.assertFalse(view.create_button.isEnabled())
                    view.close()
                    self.assertTrue(view.isVisible())
                    view.reject()
                    self.assertTrue(view.isVisible())
                    self.assertEqual(getattr(controller, method_name).call_count, 1)
                finally:
                    release.set()
                    self.wait_for_dialog(view)
                    self.assertTrue(view.create_button.isEnabled())
                    self.assertEqual(view.progress.value(), 2)
                    self.assertEqual(view.progress_label.text(), "Progress: 2/4")
                    view.close()

    def test_batch_error_restores_button(self):
        controller = Mock()
        controller.create_masks.side_effect = RuntimeError("model unavailable")
        view = PySide6GenerateMasksWindowView(None, controller, str(self.path), False)
        with patch("modules.ui.PySide6GenerateMasksWindowView.QMessageBox.critical") as show_error:
            view.create_masks()
            self.wait_for_dialog(view)
        show_error.assert_called_once_with(view, "Mask generation failed", "model unavailable")
        self.assertTrue(view.create_button.isEnabled())
        view.close()


if __name__ == "__main__":
    unittest.main()
