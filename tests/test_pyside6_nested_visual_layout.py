import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.PySide6ConceptWindowView import PySide6ConceptWindowView
from modules.ui.PySide6TimestepDistributionWindowView import PySide6TimestepDistributionWindowView
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.concept_stats import init_concept_stats
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.pyside6_i18n import current_language, set_language
from modules.util.ui.pyside6_theme import apply_theme
from modules.util.ui.PySide6UIState import PySide6UIState

from matplotlib.colors import to_rgba
from PIL import Image
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QTabWidget


class NestedVisualLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.previous_dir = Path.cwd()
        os.chdir(self.temporary_dir.name)

    def tearDown(self):
        apply_theme(self.app, "light")
        os.chdir(self.previous_dir)
        self.temporary_dir.cleanup()

    def test_timestep_chart_matches_dark_and_light_palette(self):
        apply_theme(self.app, "dark")
        config = TrainConfig.default_values()
        controller = TimestepDistributionWindowController(config)
        controller.generate_preview_data = Mock(return_value=list(range(100)))
        view = PySide6TimestepDistributionWindowView(None, controller, PySide6UIState(config))
        self.addCleanup(view.close)
        view.show()

        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                apply_theme(self.app, theme)
                self.app.processEvents()
                background = view.palette().color(QPalette.ColorRole.Window).name()
                foreground = view.palette().color(QPalette.ColorRole.WindowText).name()
                self.assertEqual(view._canvas.figure.get_facecolor(), to_rgba(background))
                self.assertEqual(view._ax.spines["bottom"].get_edgecolor(), to_rgba(foreground))

    def test_concept_preview_and_stats_fit_dialog_width(self):
        config = TrainConfig.default_values()
        concept = ConceptConfig.default_values()
        image_dir = Path(self.temporary_dir.name) / "images"
        image_dir.mkdir()
        Image.new("RGB", (64, 64), "red").save(image_dir / "synthetic.png")
        concept.path = str(image_dir)
        controller = ConceptWindowController(config, concept)
        controller.get_preview_image = Mock(return_value=(
            Image.new("RGB", (64, 64), "red"), "synthetic.png", "synthetic caption"
        ))
        controller.auto_update_concept_stats = Mock()
        view = PySide6ConceptWindowView(
            None, controller,
            PySide6UIState(concept), PySide6UIState(concept.image), PySide6UIState(concept.text),
        )
        self.addCleanup(view.close)
        tabs = view.findChild(QTabWidget)
        view.resize(900, 720)
        view.show()
        tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(view._image_layout_mode, "stacked")
        self.assertEqual(view._image_scroll.horizontalScrollBar().maximum(), 0)

        view.resize(1200, 720)
        self.app.processEvents()
        self.assertEqual(view._image_layout_mode, "side_by_side")
        self.assertEqual(view._image_scroll.horizontalScrollBar().maximum(), 0)

        view.resize(900, 720)
        tabs.setCurrentIndex(3)
        self.app.processEvents()
        self.assertEqual(tabs.widget(3).horizontalScrollBar().maximum(), 0)

    def test_empty_concept_preview_keeps_logo_transparency(self):
        config = TrainConfig.default_values()
        concept = ConceptConfig.default_values()
        controller = ConceptWindowController(config, concept)
        controller.auto_update_concept_stats = Mock()
        view = PySide6ConceptWindowView(
            None, controller,
            PySide6UIState(concept), PySide6UIState(concept.image), PySide6UIState(concept.text),
        )
        self.addCleanup(view.close)
        self.assertTrue(controller.preview_is_placeholder)
        self.assertEqual(view._filename_label.text(), "No images in this concept yet")
        self.assertTrue(view._image_label.pixmap().hasAlphaChannel())

        previous_language = current_language()
        try:
            view.show()
            set_language(self.app, "ru", persist=False)
            self.assertEqual(view.findChild(QTabWidget).tabText(0), "Общие")
            self.assertEqual(view._filename_label.text(), "В этом концепте пока нет изображений")
            controller.get_preview_image = Mock(return_value=(
                Image.new("RGB", (64, 64), "red"), "Sample", "user prompt"
            ))
            view._update_image_preview()
            self.assertEqual(view._filename_label.text(), "Sample")
            set_language(self.app, "en", persist=False)
            self.assertEqual(view._filename_label.text(), "Sample")
            self.assertEqual(view._caption_box.toPlainText(), "user prompt")
        finally:
            set_language(self.app, previous_language, persist=False)

    def test_concept_statistics_retranslate_after_language_switch(self):
        config = TrainConfig.default_values()
        concept = ConceptConfig.default_values()
        concept.concept_stats = init_concept_stats(False)
        concept.concept_stats["aspect_buckets"] = {0.5: 2, 1.0: 3, 2.0: 2}
        controller = ConceptWindowController(config, concept)
        controller.auto_update_concept_stats = Mock()
        view = PySide6ConceptWindowView(
            None, controller,
            PySide6UIState(concept), PySide6UIState(concept.image), PySide6UIState(concept.text),
        )
        self.addCleanup(view.close)
        view.show()
        view._update_concept_stats(controller)

        previous_language = current_language()
        try:
            set_language(self.app, "ru", persist=False)
            self.assertIn("соотношение", view.small_bucket_preview.text())
            set_language(self.app, "en", persist=False)
            self.assertIn("aspect", view.small_bucket_preview.text())
        finally:
            set_language(self.app, previous_language, persist=False)


if __name__ == "__main__":
    unittest.main()
