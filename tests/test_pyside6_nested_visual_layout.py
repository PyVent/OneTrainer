import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.colors import to_rgba
from PIL import Image
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QTabWidget

from modules.ui.ConceptWindowController import ConceptWindowController
from modules.ui.PySide6ConceptWindowView import PySide6ConceptWindowView
from modules.ui.PySide6TimestepDistributionWindowView import PySide6TimestepDistributionWindowView
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.pyside6_theme import apply_theme


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


if __name__ == "__main__":
    unittest.main()
