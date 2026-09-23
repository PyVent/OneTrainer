import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView


class CaptionUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name)

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
        caption_controller.create_captions.assert_called_once_with(
            model_name="Blip", path=str(self.path), initial_caption="",
            caption_prefix="prefix", caption_postfix="",
            mode_str="Create if absent", include_subdirectories=False,
        )
        caption_view.close()


if __name__ == "__main__":
    unittest.main()
