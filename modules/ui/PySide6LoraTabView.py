from modules.ui.BaseLoraTabView import BaseLoraTabView
from modules.ui.LoraTabController import LoraTabController
from modules.util.enum.ModelType import PeftType
from modules.util.ui import pyside6_components

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PySide6LoraTabView(BaseLoraTabView, QWidget):

    def __init__(self, master, controller: LoraTabController, ui_state):
        QWidget.__init__(self, master)
        BaseLoraTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self.scroll_area: QScrollArea | None = None
        self.options_frame = None
        self._options_grid: QGridLayout | None = None
        self._option_cards: list[QWidget] = []
        self._option_columns = 0
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_area is not None:
            self.scroll_area.viewport().removeEventFilter(self)
            self.scroll_area.hide()
            self.scroll_area.deleteLater()
            self.options_frame = None
            self._options_grid = None
            self._option_cards = []

        self.scroll_area, self.scroll_frame = pyside6_components.scrollable_frame(self)
        outer = pyside6_components._layout(self)
        outer.setRowStretch(0, 1)
        outer.setColumnStretch(0, 1)
        outer.addWidget(self.scroll_area, 0, 0)
        self.scroll_area.widget().layout().setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        lo = pyside6_components._layout(self.scroll_frame)
        lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(2, 2)
        self.build(self.scroll_frame, self.controller, self.ui_state, self.setup_lora)
        pyside6_components._pack_form(self.scroll_frame)
        self.scroll_area.viewport().installEventFilter(self)
        self._reflow_options()

    def setup_lora(self, peft_type: PeftType):
        if self.options_frame is not None:
            self.options_frame.hide()
            self.options_frame.deleteLater()

        self.options_frame = QGroupBox(f"{peft_type.value} settings", self.scroll_frame)
        pyside6_components._layout(self.scroll_frame).addWidget(self.options_frame, 1, 0, 1, 3)
        self.options_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        staging = QWidget(self.options_frame)
        self.build_lora_options(staging, self.controller, self.ui_state, peft_type)
        source_layout = pyside6_components._layout(staging)
        rows: dict[int, dict[int, QWidget]] = {}
        for index in range(source_layout.count()):
            row, column, _, _ = source_layout.getItemPosition(index)
            widget = source_layout.itemAt(index).widget()
            if widget is not None:
                rows.setdefault(row, {})[column] = widget
        while source_layout.count():
            source_layout.takeAt(0)

        outer = QVBoxLayout(self.options_frame)
        outer.setContentsMargins(12, 16, 12, 12)
        self._options_grid = QGridLayout()
        self._options_grid.setHorizontalSpacing(16)
        self._options_grid.setVerticalSpacing(10)
        outer.addLayout(self._options_grid)

        self._option_cards = []
        for columns in rows.values():
            for label_column, control_column in ((0, 1), (3, 4)):
                if label_column not in columns and control_column not in columns:
                    continue
                card = QFrame(self.options_frame)
                card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(0, 0, 0, 0)
                card_layout.setSpacing(4)
                label = columns.get(label_column)
                control = columns.get(control_column)
                if label is not None:
                    if isinstance(label, QLabel):
                        label.setWordWrap(True)
                    card_layout.addWidget(label)
                if control is not None:
                    card_layout.addWidget(control)
                self._option_cards.append(card)

        staging.deleteLater()
        self._option_columns = 0
        self._reflow_options()

    def _reflow_options(self):
        if self._options_grid is None or self.scroll_area is None:
            return
        columns = 2 if self.scroll_area.viewport().width() >= 760 else 1
        if columns == self._option_columns:
            return
        self._option_columns = columns
        while self._options_grid.count():
            self._options_grid.takeAt(0)
        for index, card in enumerate(self._option_cards):
            self._options_grid.addWidget(card, index // columns, index % columns)
        self._options_grid.setColumnStretch(0, 1)
        self._options_grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def eventFilter(self, watched, event):
        if self.scroll_area is not None and watched is self.scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self._reflow_options()
        return super().eventFilter(watched, event)
