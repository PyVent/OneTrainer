from modules.ui.BaseVideoToolUIView import BaseVideoToolUIView
from modules.ui.VideoToolUIController import VideoToolUIController
from modules.util.image_util import load_image
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QWidget,
)

_PAD = pyside6_components.PAD


class PySide6VideoToolUIView(BaseVideoToolUIView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: VideoToolUIController):
        QDialog.__init__(self, parent)
        BaseVideoToolUIView.__init__(self, pyside6_components)

        self.controller = controller
        self._status_box: QTextEdit | None = None
        self._preview_label: QLabel | None = None
        self._preview_caption_label: QLabel | None = None

        self.ui_state = PySide6UIState(controller.args)

        self.setWindowTitle("Video Tools")
        self.resize(700, 750)

        outer = QGridLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.setRowStretch(0, 1)
        outer.setRowStretch(1, 0)

        tabs = QTabWidget(self)
        outer.addWidget(tabs, 0, 0)

        for name, build_fn in [
            ("extract clips", self.build_clip_extract_tab),
            ("extract images", self.build_image_extract_tab),
            ("download", self.build_video_download_tab),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            frame = QWidget()
            scroll.setWidget(frame)
            lo = pyside6_components._layout(frame)
            lo.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
            lo.setColumnMinimumWidth(0, 120)
            lo.setColumnStretch(3, 1)
            build_fn(frame, controller, self.ui_state)
            if name != "download":
                self._arrange_time_range(frame)
            lo.setRowStretch(lo.rowCount(), 1)
            tabs.addTab(scroll, name)

        outer.addWidget(self._build_status_bar(), 1, 0)

    @staticmethod
    def _arrange_time_range(frame: QWidget):
        """The shared builder places both endpoints in one legacy grid cell."""
        layout = pyside6_components._layout(frame)
        endpoints = [
            layout.itemAt(index).widget()
            for index in range(layout.count())
            if layout.getItemPosition(index)[:2] == (1, 1)
            and isinstance(layout.itemAt(index).widget(), QLineEdit)
        ]
        if len(endpoints) != 2:
            return
        range_frame = QWidget(frame)
        range_layout = QHBoxLayout(range_frame)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(8)
        for endpoint in endpoints:
            layout.removeWidget(endpoint)
        range_layout.addWidget(endpoints[0], 1)
        range_layout.addWidget(QLabel("to", range_frame))
        range_layout.addWidget(endpoints[1], 1)
        layout.addWidget(range_frame, 1, 1)

    def _build_status_bar(self):
        frame = QWidget(self)
        lo = QGridLayout(frame)
        lo.setColumnMinimumWidth(0, 160)
        lo.setColumnStretch(2, 1)

        self._preview_label = QLabel(frame)
        self._preview_label.setFixedSize(150, 150)
        preview = load_image(str(Path(__file__).resolve().parents[2] / "resources/icons/icon.png"), 'RGB')
        preview.thumbnail((150, 150))
        self._preview_label.setPixmap(
            QPixmap.fromImage(ImageQt(preview.convert("RGBA"))).scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self._preview_caption_label = QLabel("Preview image", frame)
        self._preview_caption_label.setWordWrap(True)

        preview_col = QWidget(frame)
        preview_lo = QGridLayout(preview_col)
        preview_lo.setContentsMargins(0, 0, 0, 0)
        preview_lo.addWidget(self._preview_label, 0, 0, Qt.AlignTop)
        preview_lo.addWidget(self._preview_caption_label, 1, 0, Qt.AlignTop)
        lo.addWidget(preview_col, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        self._status_box = QTextEdit(frame)
        self._status_box.setReadOnly(True)
        self._status_box.setFixedHeight(160)
        self._status_box.setMinimumWidth(300)
        self._status_box.setPlainText("Current status")
        lo.addWidget(self._status_box, 0, 1, Qt.AlignTop)

        return frame

    # --- abstract method implementations ---

    def _create_textbox(self, master, row, col, width, height, ui_state, var_name):
        var = ui_state.get_var(var_name)
        widget = QTextEdit(master)
        widget.setFixedHeight(height)
        widget.setMinimumWidth(width)
        widget.setPlainText(var.get())
        pyside6_components._add(
            pyside6_components._layout(master), widget, row, col, sticky="w", rowspan=2
        )
        widget.textChanged.connect(lambda: var.set(widget.toPlainText()))
        def sync_from_state(value):
            value = str(value)
            if widget.toPlainText() != value:
                with QSignalBlocker(widget):
                    widget.setPlainText(value)

        var.subscribe(sync_from_state, owner=widget)
        return widget

    def schedule_on_main_thread(self, fn):
        QTimer.singleShot(0, self, fn)

    def update_status(self, status_text: str):
        # Called from the video tool's worker thread — defer to main thread
        self.schedule_on_main_thread(lambda: self._status_box.append(status_text))

    def clear_status(self):
        self._status_box.clear()

    def update_preview(self, preview_image, label_text: str):
        # Called from the video tool's worker thread — defer to main thread
        image = ImageQt(preview_image.convert("RGBA")).copy()
        self.schedule_on_main_thread(lambda: self._do_update_preview(image, label_text))

    def _do_update_preview(self, image: QImage, label_text: str):
        self._preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._preview_caption_label.setText(label_text)
