from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseAdditionalEmbeddingsTabView import BaseAdditionalEmbeddingsTabView, BaseEmbeddingWidgetView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import Qt
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


class PySide6AdditionalEmbeddingsTabView(PySide6ConfigListView, BaseAdditionalEmbeddingsTabView):

    def __init__(self, master, controller: AdditionalEmbeddingsTabController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="additional_embeddings",
            enable_key="train",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
            show_toggle_button=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6EmbeddingWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class PySide6EmbeddingWidgetView(BaseEmbeddingWidgetView, QWidget):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        QWidget.__init__(self, master)
        BaseEmbeddingWidgetView.__init__(self, pyside6_components)

        self.element = element
        ui_state = PySide6UIState(element)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        # Let the shared builder create and bind every control, then move the
        # existing Qt widgets into a compact layout. No field rules are copied.
        top_frame = QWidget(self)
        bottom_frame = QWidget(self)
        self.build_content(top_frame, bottom_frame, ui_state, i, save_command, remove_command, clone_command, controller)

        top = self._take_row(top_frame)
        bottom = self._take_row(bottom_frame)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(0)
        group = QGroupBox("Embedding", self)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        root.addWidget(group)
        root.addStretch(1)

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(12, 12, 12, 12)
        group_layout.setSpacing(10)

        actions = QHBoxLayout()
        actions.addStretch(1)
        top[1].setAccessibleName("Clone embedding")
        top[0].setAccessibleName("Remove embedding")
        actions.addWidget(top[1])
        actions.addWidget(top[0])
        group_layout.addLayout(actions)

        self._field_grids: list[tuple[QGridLayout, list[QWidget]]] = []
        self._add_fields(group_layout, "Source and tokens", top, ((2, 3), (4, 5), (6, 7)))
        self._add_fields(group_layout, "Training", bottom, ((0, 1), (2, 3), (4, 5), (6, 7)))

        self._column_count = 0
        self._reflow()
        top_frame.deleteLater()
        bottom_frame.deleteLater()

    @staticmethod
    def _take_row(frame: QWidget) -> dict[int, QWidget]:
        layout = frame.layout()
        widgets = {
            layout.getItemPosition(index)[1]: layout.itemAt(index).widget()
            for index in range(layout.count())
        }
        while layout.count():
            layout.takeAt(0)
        return widgets

    def _add_fields(self, parent_layout: QVBoxLayout, heading: str, widgets: dict[int, QWidget], pairs):
        title = QLabel(heading, self)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        parent_layout.addWidget(title)

        field_grid = QGridLayout()
        field_grid.setHorizontalSpacing(16)
        field_grid.setVerticalSpacing(8)
        cards = []
        for label_column, control_column in pairs:
            card = QFrame(self)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setSpacing(4)
            label = widgets[label_column]
            if isinstance(label, QLabel):
                label.setWordWrap(True)
            card_layout.addWidget(label)
            card_layout.addWidget(widgets[control_column])
            cards.append(card)
        parent_layout.addLayout(field_grid)
        self._field_grids.append((field_grid, cards))

    def _reflow(self):
        columns = 2 if self.width() >= 760 else 1
        if columns == self._column_count:
            return
        self._column_count = columns
        for grid, cards in self._field_grids:
            while grid.count():
                grid.takeAt(0)
            for index, card in enumerate(cards):
                grid.addWidget(card, index // columns, index % columns)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def place_in_list(self):
        layout = pyside6_components._layout(self.parent())
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self, getattr(self, 'visible_index', self.i), 0, Qt.AlignmentFlag.AlignTop)
        self.show()

    def destroy(self):
        self.deleteLater()
