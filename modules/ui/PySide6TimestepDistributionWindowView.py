from modules.ui.BaseTimestepDistributionWindowView import BaseTimestepDistributionWindowView
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.util.ui import pyside6_components

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QEvent
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QSizePolicy, QWidget


class PySide6TimestepDistributionWindowView(BaseTimestepDistributionWindowView, QDialog):
    WIDE_VIEWPORT_WIDTH = 850

    def __init__(self, parent, controller: TimestepDistributionWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseTimestepDistributionWindowView.__init__(self, pyside6_components)

        # delete on close so entry widgets and the field validators they register globally are freed, not leaked
        self.finished.connect(self.deleteLater)

        self.setWindowTitle("Timestep Distribution")
        self.resize(900, 600)
        self._controller = controller

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll, frame = pyside6_components.scrollable_frame(self)
        self._scroll = scroll
        self._content_layout = pyside6_components._layout(frame)

        self._fields_frame = QWidget(frame)
        fields_layout = pyside6_components._layout(self._fields_frame)
        fields_layout.setColumnStretch(1, 1)
        self.build_content(self._fields_frame, controller, ui_state)
        self._fields_layout = fields_layout
        self._field_pairs = tuple(
            (fields_layout.itemAtPosition(row, 0).widget(), fields_layout.itemAtPosition(row, 1).widget())
            for row in range(7)
        )
        self._fields_stacked = None

        self._preview_frame = QWidget(frame)
        preview_layout = pyside6_components._layout(self._preview_frame)
        preview_layout.setColumnStretch(0, 1)
        preview_layout.setRowStretch(0, 1)

        fig, self._ax = plt.subplots()
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setMinimumSize(320, 240)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        preview_layout.addWidget(self._canvas, 0, 0)
        self.finished.connect(lambda _: plt.close(fig))

        update_btn = QPushButton("Update Preview", self._preview_frame)
        update_btn.clicked.connect(self._update_preview)
        preview_layout.addWidget(update_btn, 1, 0)

        self._layout_mode = None
        scroll.viewport().installEventFilter(self)
        self._reflow_content(scroll.viewport().width())
        self._update_preview()

        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("OK", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)

    def eventFilter(self, watched, event):
        if watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._reflow_content(event.size().width())
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "_canvas"):
            self._apply_chart_palette()
            self._canvas.draw_idle()

    def _reflow_content(self, width: int):
        self._reflow_fields(width)
        mode = "wide" if width >= self.WIDE_VIEWPORT_WIDTH else "compact"
        layout = self._content_layout
        if mode != self._layout_mode:
            layout.removeWidget(self._fields_frame)
            layout.removeWidget(self._preview_frame)
            if mode == "wide":
                layout.setColumnStretch(0, 0)
                layout.setColumnStretch(1, 1)
                layout.addWidget(self._fields_frame, 0, 0)
                layout.addWidget(self._preview_frame, 0, 1)
            else:
                layout.setColumnStretch(0, 1)
                layout.setColumnStretch(1, 0)
                layout.addWidget(self._fields_frame, 0, 0)
                layout.addWidget(self._preview_frame, 1, 0)
            self._layout_mode = mode

        # QScrollArea may retain the old content width after the fields switch
        # from two columns to one. Bring it back to the viewport immediately.
        self._fields_layout.activate()
        layout.activate()
        content = self._scroll.widget()
        content.resize(max(width, content.minimumSizeHint().width()), content.height())

    def _reflow_fields(self, width: int):
        stacked = width < 600
        if stacked == self._fields_stacked:
            return

        layout = self._fields_layout
        for label, control in self._field_pairs:
            layout.removeWidget(label)
            layout.removeWidget(control)
        layout.setColumnStretch(0, 1 if stacked else 0)
        layout.setColumnStretch(1, 0 if stacked else 1)
        for row, (label, control) in enumerate(self._field_pairs):
            if stacked:
                layout.addWidget(label, row * 2, 0)
                layout.addWidget(control, row * 2 + 1, 0)
            else:
                layout.addWidget(label, row, 0)
                layout.addWidget(control, row, 1)
        self._fields_stacked = stacked


    def _update_preview(self):
        self._ax.cla()
        self._ax.hist(self._controller.generate_preview_data(), bins=1000, range=(0, 999))
        self._apply_chart_palette()
        self._canvas.draw()

    def _apply_chart_palette(self):
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Window).name()
        foreground = palette.color(QPalette.ColorRole.WindowText).name()
        accent = palette.color(QPalette.ColorRole.Highlight).name()
        self._canvas.figure.set_facecolor(background)
        self._ax.set_facecolor(background)
        for patch in self._ax.patches:
            patch.set_facecolor(accent)
        for spine in self._ax.spines.values():
            spine.set_color(foreground)
        self._ax.tick_params(colors=foreground)
        self._ax.xaxis.label.set_color(foreground)
        self._ax.yaxis.label.set_color(foreground)
