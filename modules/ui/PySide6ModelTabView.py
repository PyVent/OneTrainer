from modules.ui.BaseModelTabView import BaseModelTabView
from modules.ui.ModelTabController import ModelTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PySide6ModelTabView(BaseModelTabView, QWidget, metaclass=QtABCMeta):
    """Present the shared model fields as responsive Qt sections."""

    def __init__(self, master, controller: ModelTabController, ui_state):
        QWidget.__init__(self, master)
        BaseModelTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self._sections: list[tuple[QGridLayout, list[QWidget]]] = []
        self._column_count = 0
        self.refresh_ui()

    def _make_svd_frames(self, parent, row: int):
        svd_label_frame = QWidget(parent)
        pyside6_components._layout(parent).addWidget(svd_label_frame, row, 3)
        svd_entry_frame = QWidget(parent)
        pyside6_components._layout(parent).addWidget(svd_entry_frame, row, 4)
        return svd_label_frame, svd_entry_frame

    @staticmethod
    def _take_fields(frame: QWidget) -> list[tuple[int, QWidget | None, QWidget | None]]:
        """Take each field pair without rebuilding or rebinding its widgets."""
        layout = frame.layout()
        rows: dict[int, dict[int, QWidget]] = {}
        for index in range(layout.count()):
            item = layout.itemAt(index)
            row, column, _, _ = layout.getItemPosition(index)
            if item.widget() is not None:
                rows.setdefault(row, {})[column] = item.widget()

        while layout.count():
            layout.takeAt(0)

        fields = []
        for row, columns in sorted(rows.items()):
            for label_column, control_column in ((0, 1), (3, 4)):
                if label_column in columns or control_column in columns:
                    fields.append((row, columns.get(label_column), columns.get(control_column)))
            unexpected = set(columns) - {0, 1, 3, 4}
            if unexpected:
                raise RuntimeError(f"Unexpected Model field columns: {sorted(unexpected)}")
        return fields

    @staticmethod
    def _field_card(label: QWidget | None, control: QWidget | None, parent: QWidget) -> QWidget:
        card = QFrame(parent)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if label is not None and control is not None and not isinstance(label, QLabel):
            # SVDQuant supplies two small grids of aligned labels and controls.
            layout = QHBoxLayout(card)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(12)
            layout.addWidget(label)
            layout.addWidget(control, 1)
        else:
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)
            if label is not None:
                if isinstance(label, QLabel):
                    label.setWordWrap(True)
                layout.addWidget(label)
            if control is not None:
                layout.addWidget(control)
        return card

    def _add_section(self, container: QWidget, title: str, fields):
        if not fields:
            return
        group = QGroupBox(title, container)
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 16, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        cards = [self._field_card(label, control, group) for _, label, control in fields]
        self._sections.append((grid, cards))
        container.layout().addWidget(group)

    def _reflow(self):
        if self.scroll_frame is None or not self._sections:
            return
        columns = 2 if self.scroll_frame.viewport().width() >= 1080 else 1
        if columns == self._column_count:
            return
        self._column_count = columns
        for grid, cards in self._sections:
            while grid.count():
                grid.takeAt(0)
            for index, card in enumerate(cards):
                grid.addWidget(card, index // columns, index % columns)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def eventFilter(self, watched, event):
        if self.scroll_frame is not None and watched is self.scroll_frame.viewport() and event.type() == QEvent.Type.Resize:
            self._reflow()
        return super().eventFilter(watched, event)

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.viewport().removeEventFilter(self)
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        self.scroll_frame = scroll
        self._sections = []
        self._column_count = 0

        # BaseModelTabView remains the source of fields, options and UIState
        # bindings. Only their Qt ownership and layout change here.
        staging = QWidget(frame)
        self.build_content(staging, self.controller, self.ui_state)
        fields = self._take_fields(staging)
        last_row = max(row for row, _, _ in fields)

        content_layout = QVBoxLayout(frame)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(14)
        self._add_section(frame, "Source and access", [field for field in fields if field[0] < 3])
        self._add_section(frame, "Model components and precision", [field for field in fields if 3 <= field[0] < last_row - 1])
        self._add_section(frame, "Output", [field for field in fields if field[0] >= last_row - 1])
        content_layout.addStretch(1)
        staging.deleteLater()

        scroll.viewport().installEventFilter(self)
        self._reflow()
