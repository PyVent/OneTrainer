import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.PySide6ConceptWindowView import PySide6ConceptWindowView
from modules.util.caption_util import caption_key
from modules.util.concept_stats import folder_scan, init_concept_stats
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.pyside6_i18n import current_language, set_language
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDoubleSpinBox


class ConceptEditorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 64), "red").save(self.root / "a.png")
        (self.root / "a.txt").write_text("red, blue, green\nA red picture.", encoding="utf-8")
        self.config = TrainConfig.default_values()
        self.concept = ConceptTabController(self.config).create_new_element()
        self.concept.path = str(self.root)
        self.concept.seed = 9876543210123
        self.concept.image.random_hue_max_strength = 0.8
        self.concept.image.enable_fixed_hue = True
        self.states = [PySide6UIState(obj) for obj in (self.concept, self.concept.image, self.concept.text)]
        self.language = current_language()
        set_language(self.app, "en", persist=False)
        self.addCleanup(set_language, self.app, self.language, persist=False)

    def editor(self):
        controller = ConceptWindowController(self.config, self.concept)
        view = PySide6ConceptWindowView(None, controller, *self.states)
        self.addCleanup(view.close)
        return view

    def wait_for_preview(self, view):
        view._update_image_preview()
        deadline = time.monotonic() + 10
        while view._preview_busy and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertFalse(view._preview_busy, "Preview did not finish")

    def test_new_defaults_and_old_concepts_load_through_tab_controller(self):
        self.assertEqual(self.concept.text.caption_mode, "all")
        self.assertEqual(self.concept.text.caption_format, "auto")
        self.concept.from_dict({"__version": 2, "text": {"enable_tag_shuffling": True}})
        self.assertEqual(self.concept.text.caption_mode, "random")
        self.assertEqual(self.concept.text.caption_format, "tags")
        self.assertTrue(self.concept.text.enable_tag_shuffling)

    def test_save_flushes_pending_edits_and_cancel_preserves_saved_state(self):
        view = self.editor()
        self.wait_for_preview(view)
        hue = view._controls["image_mode_hue"].parent().findChild(QDoubleSpinBox)
        self.assertEqual(hue.value(), 0.8)
        self.assertEqual(view._controls["seed"].text(), "9876543210123")
        view._controls["name"].setText("Edited concept")
        view._controls["image_variations"].lineEdit().setText("3")
        view._caption_override.setCurrentIndex(2)
        self.assertEqual(self.concept.name, "")
        self.assertEqual(self.concept.image_variations, 1)
        self.assertEqual(self.concept.text.caption_overrides, {})
        view._ok()
        self.assertEqual(view.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.concept.name, "Edited concept")
        self.assertEqual(self.concept.image_variations, 3)
        self.assertEqual(self.concept.text.caption_overrides[caption_key("red, blue, green")], "text")
        self.assertEqual(self.concept.seed, 9876543210123)
        self.assertEqual(self.concept.image.random_hue_max_strength, 0.8)
        # The surrounding concept list must still edit the saved nested objects.
        self.states[2].get_var("caption_mode").set("random")
        self.assertEqual(self.concept.text.caption_mode, "random")
        saved = self.concept.to_dict()
        cancelled = self.editor()
        self.wait_for_preview(cancelled)
        cancelled._controls["name"].setText("Discard me")
        cancelled._controls["image_variations"].setValue(9)
        cancelled._caption_override.setCurrentIndex(1)
        cancelled.reject()
        self.assertEqual(self.concept.to_dict(), saved)

    def test_opening_settings_does_not_read_images_or_enumerate_dataset(self):
        with patch.object(ConceptWindowController, "get_preview_image") as preview:
            view = self.editor()
            view.show()
            QTest.qWait(350)
            preview.assert_not_called()

    def test_preview_discovers_only_the_requested_image(self):
        controller = ConceptWindowController(self.config, self.concept)
        self.addCleanup(controller.close_preview)
        visited = []

        def paths(*_):
            for index in range(10000):
                visited.append(index)
                yield self.root / "a.png"

        controller._iter_preview_paths = paths
        controller.get_preview_image(0, False)
        self.assertEqual(visited, [0])
        controller.get_preview_image(0, True)
        self.assertEqual(visited, [0])
        controller.get_preview_image(1, False)
        self.assertEqual(visited, [0, 1])

    def test_invalid_text_setting_prevents_save(self):
        view = self.editor()
        view._controls["tag_delimiter"].setText("")
        view._ok()
        self.assertFalse(view._closed)
        self.assertEqual(self.concept.text.tag_delimiter, ",")

    def test_scan_counts_examples_and_discards_results_on_cancel(self):
        view = self.editor()
        view.controller.get_concept_stats(view, True, 10)
        self.app.processEvents()
        stats = view.controller.concept.concept_stats
        self.assertTrue(stats["scan_complete"])
        self.assertEqual(stats["image_count"], 1)
        self.assertEqual(stats["image_caption_examples"], 2)
        self.assertIn("2 samples", view._sampling_summary.text())
        self.assertEqual(self.concept.concept_stats, {})

        view.ui_state.get_var("enabled").set(False)
        view._update_sampling_summary()
        self.assertIn("0 samples", view._sampling_summary.text())
        view.controller.cancel_scan_flag.set()
        view.controller.get_concept_stats(view, False, 10)
        self.app.processEvents()
        self.assertTrue(view.controller.concept.concept_stats["force_cancelled"])
        self.assertIn("incomplete", view._scan_status.text())
        view.reject()
        view._enable_scan_buttons()  # A queued worker callback after closing is harmless.
        self.assertEqual(self.concept.concept_stats, {})

    def test_scan_publishes_partial_counts_inside_one_directory(self):
        for index in range(10):
            (self.root / f"{index}.png").touch()
        times = iter(i * 0.1 for i in range(1000))
        snapshots = []
        with patch("modules.util.concept_stats.time.perf_counter", side_effect=lambda: next(times)):
            stats = folder_scan(str(self.root), init_concept_stats(False), False, self.concept, 0, 9999,
                                threading.Event(), lambda stats: snapshots.append(stats["image_count"]))
        self.assertTrue(any(0 < count < stats["image_count"] for count in snapshots))


if __name__ == "__main__":
    unittest.main()
