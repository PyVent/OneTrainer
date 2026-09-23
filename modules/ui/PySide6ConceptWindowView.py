import threading
from pathlib import Path

from modules.ui.BaseConceptWindowView import BaseConceptWindowView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_i18n import set_localized_text

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QWidget,
)


class NonZoomingFigureCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        event.ignore()


class PySide6ConceptWindowView(BaseConceptWindowView, QDialog):
    def __init__(
            self,
            parent,
            controller: ConceptWindowController,
            ui_state,
            image_ui_state,
            text_ui_state,
    ):
        QDialog.__init__(self, parent)
        BaseConceptWindowView.__init__(self, pyside6_components)

        self.controller = controller
        self.image_preview_file_index = 0
        self._preview_augmentations = True
        self.bucket_fig = None
        self._image_layout_mode = None

        self.setWindowTitle("Concept")
        self.resize(800, 700)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        tabs = QTabWidget(self)
        outer.addWidget(tabs, 0, 0)

        _pad = pyside6_components.PAD

        # --- general tab ---
        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_frame = QWidget()
        gen_scroll.setWidget(gen_frame)
        pyside6_components._layout(gen_frame).setContentsMargins(_pad, _pad, _pad, _pad)
        pyside6_components._layout(gen_frame).setColumnStretch(1, 1)
        pyside6_components._layout(gen_frame).setColumnStretch(2, 1)
        self.build_general_tab(gen_frame, controller, ui_state, text_ui_state)
        pyside6_components._pack_form(gen_frame)
        tabs.addTab(gen_scroll, "general")

        # --- image augmentation tab ---
        img_scroll = QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_outer = QWidget()
        img_scroll.setWidget(img_outer)
        lo_img_outer = pyside6_components._layout(img_outer)
        lo_img_outer.setContentsMargins(_pad, _pad, _pad, _pad)
        lo_img_outer.setColumnStretch(0, 1)

        # form in its own widget so the preview panel can't affect row heights
        img_form = QWidget(img_outer)
        img_form_lo = pyside6_components._layout(img_form)
        img_form_lo.setColumnStretch(3, 1)
        self.build_image_augmentation_tab(img_form, controller, image_ui_state)
        pyside6_components._pack_form(img_form)
        lo_img_outer.addWidget(img_form, 0, 0, Qt.AlignTop)

        # preview panel alongside the form
        image_preview, filename_preview, caption_preview = controller.get_preview_image(
            self.image_preview_file_index, self._preview_augmentations
        )
        preview_panel = QWidget(img_outer)
        pb_lo = QGridLayout(preview_panel)

        self._image_label = QLabel(preview_panel)
        self._image_label.setObjectName("conceptPreview")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setFixedSize(300, 300)
        self._image_label.setPixmap(self._preview_pixmap(image_preview))
        pb_lo.addWidget(self._image_label, 0, 0, 1, 3)

        prev_btn = QPushButton("Previous", preview_panel)
        prev_btn.setToolTip("Show previous preview image")
        prev_btn.clicked.connect(self._prev_image_preview)
        update_btn = QPushButton("Update Preview", preview_panel)
        update_btn.clicked.connect(self._update_image_preview)
        next_btn = QPushButton("Next", preview_panel)
        next_btn.setToolTip("Show next preview image")
        next_btn.clicked.connect(self._next_image_preview)
        self._aug_checkbox = QCheckBox("Show Augmentations", preview_panel)
        self._aug_checkbox.setChecked(True)
        self._aug_checkbox.toggled.connect(lambda checked: self._on_aug_toggle(checked))
        pb_lo.addWidget(prev_btn, 1, 0)
        pb_lo.addWidget(update_btn, 1, 1)
        pb_lo.addWidget(next_btn, 1, 2)
        pb_lo.addWidget(self._aug_checkbox, 2, 0, 1, 3)

        self._filename_label = QLabel(preview_panel)
        self._set_preview_filename(filename_preview)
        self._filename_label.setWordWrap(True)
        self._filename_label.setFixedWidth(300)
        pb_lo.addWidget(self._filename_label, 3, 0, 1, 3)

        self._caption_box = QTextEdit(preview_panel)
        self._caption_box.setReadOnly(True)
        self._caption_box.setPlainText(caption_preview)
        self._caption_box.setFixedSize(300, 150)
        pb_lo.addWidget(self._caption_box, 4, 0, 1, 3)

        lo_img_outer.addWidget(preview_panel, 0, 1, Qt.AlignTop)
        self._image_scroll = img_scroll
        self._image_layout = lo_img_outer
        self._image_form = img_form
        self._image_preview_panel = preview_panel
        img_scroll.viewport().installEventFilter(self)
        self._reflow_image_preview(img_scroll.viewport().width())
        tabs.addTab(img_scroll, "image augmentation")

        # --- text augmentation tab ---
        text_scroll = QScrollArea()
        text_scroll.setWidgetResizable(True)
        text_frame = QWidget()
        text_scroll.setWidget(text_frame)
        pyside6_components._layout(text_frame).setContentsMargins(_pad, _pad, _pad, _pad)
        pyside6_components._layout(text_frame).setColumnStretch(1, 1)
        pyside6_components._layout(text_frame).setColumnStretch(2, 1)
        pyside6_components._layout(text_frame).setColumnStretch(3, 1)
        self.build_text_augmentation_tab(text_frame, controller, text_ui_state)
        pyside6_components._pack_form(text_frame)
        tabs.addTab(text_scroll, "text augmentation")

        # --- statistics tab ---
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_frame = QWidget()
        stats_scroll.setWidget(stats_frame)
        stats_lo = pyside6_components._layout(stats_frame)
        stats_lo.setContentsMargins(_pad, _pad, _pad, _pad)
        stats_lo.setColumnMinimumWidth(0, 150)
        stats_lo.setColumnMinimumWidth(1, 150)
        stats_lo.setColumnMinimumWidth(2, 150)
        stats_lo.setColumnMinimumWidth(3, 150)
        self.build_concept_stats_tab(stats_frame, controller)
        stats_lo.setVerticalSpacing(10)
        for label in stats_frame.findChildren(QLabel):
            if label.text().startswith("\n"):
                label.setText(label.text().lstrip("\n"))
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        plt.set_loglevel('WARNING')
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7, 3))
        self.canvas = NonZoomingFigureCanvas(self.bucket_fig)
        self.bucket_fig.tight_layout()
        self.bucket_fig.subplots_adjust(bottom=0.15)

        palette = self.palette()
        self.text_color = palette.text().color().name()
        background_color = palette.window().color().name()
        self.bucket_fig.set_facecolor(background_color)
        self.bucket_ax.set_facecolor(background_color)
        self.bucket_ax.spines['bottom'].set_color(self.text_color)
        self.bucket_ax.spines['left'].set_color(self.text_color)
        self.bucket_ax.spines['top'].set_visible(False)
        self.bucket_ax.spines['right'].set_color(self.text_color)
        self.bucket_ax.tick_params(axis='x', colors=self.text_color, which="both")
        self.bucket_ax.tick_params(axis='y', colors=self.text_color, which="both")
        self.bucket_ax.xaxis.label.set_color(self.text_color)
        self.bucket_ax.yaxis.label.set_color(self.text_color)

        stats_lo.addWidget(self.canvas, 19, 0, 2, 4)

        tabs.addTab(stats_scroll, "statistics")

        ok = QPushButton("Done", self)
        ok.clicked.connect(self._ok)
        outer.addWidget(ok, 1, 0)

        #automatic concept scan
        self.scan_thread = threading.Thread(target=controller.auto_update_concept_stats, args=[self], daemon=True)
        self.scan_thread.start()

    def eventFilter(self, watched, event):
        if watched is self._image_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._reflow_image_preview(event.size().width())
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.LanguageChange and self.bucket_fig is not None:
            # Matplotlib labels are outside Qt's widget translation tree.
            stats = self.controller.concept.concept_stats
            if isinstance(stats, dict) and "aspect_buckets" in stats:
                self._update_concept_stats(self.controller)

    def _reflow_image_preview(self, width: int):
        mode = "side_by_side" if width >= 1000 else "stacked"
        if mode == self._image_layout_mode:
            return
        layout = self._image_layout
        layout.removeWidget(self._image_form)
        layout.removeWidget(self._image_preview_panel)
        if mode == "side_by_side":
            layout.addWidget(self._image_form, 0, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._image_preview_panel, 0, 1, Qt.AlignmentFlag.AlignTop)
            layout.setColumnStretch(1, 0)
        else:
            layout.addWidget(self._image_form, 0, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._image_preview_panel, 1, 0, Qt.AlignmentFlag.AlignHCenter)
            layout.setColumnStretch(1, 0)
        self._image_layout_mode = mode


    def _on_aug_toggle(self, checked: bool):
        self._preview_augmentations = checked
        self._update_image_preview()

    def _prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self._update_image_preview()

    def _next_image_preview(self):
        self.image_preview_file_index += 1
        self._update_image_preview()

    def _update_image_preview(self):
        image_preview, filename_preview, caption_preview = self.controller.get_preview_image(
            self.image_preview_file_index, self._preview_augmentations
        )
        self._image_label.setPixmap(self._preview_pixmap(image_preview))
        self._set_preview_filename(filename_preview)
        self._caption_box.setPlainText(caption_preview)

    def _set_preview_filename(self, filename: str):
        if filename == "No images in this concept yet":
            self._filename_label.setProperty("_onetrainer_i18n_keep_native", False)
            set_localized_text(self._filename_label, filename)
        else:
            # A real filename belongs to the user and must remain verbatim.
            self._filename_label.setProperty("_onetrainer_i18n_keep_native", True)
            self._filename_label.setProperty("_onetrainer_i18n_template", None)
            self._filename_label.setText(filename)

    def _preview_pixmap(self, image):
        if getattr(self.controller, "preview_is_placeholder", False):
            icon = Path(__file__).resolve().parents[2] / "resources" / "icons" / "icon.png"
            return QPixmap(str(icon)).scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(ImageQt(image.convert("RGBA"))).scaled(
            300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def _cleanup(self):
        # stop the background scan thread (reuses the Abort Scan mechanism) so it
        # stops touching this window's widgets, and release the matplotlib figure
        self.controller.cancel_scan_flag.set()
        if self.bucket_fig is not None:
            plt.close(self.bucket_fig)
            self.bucket_fig = None

    def closeEvent(self, event):
        # also reached when the window is closed via the OS X button
        self._cleanup()
        super().closeEvent(event)

    def _ok(self):
        self._cleanup()
        self.accept()
