from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseAdditionalEmbeddingsTabView import BaseAdditionalEmbeddingsTabView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.util import path_util
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
            add_button_text="Add Embedding",
            is_full_width=True,
            show_toggle_button=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6EmbeddingWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class PySide6EmbeddingWidgetView(QWidget):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        super().__init__(master)

        self.element = element
        self.ui_state = PySide6UIState(element)
        self.i = i
        self.save_command = save_command
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

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
        copy = QPushButton("Copy", group)
        copy.setObjectName("copyAction")
        copy.setAccessibleName("Clone embedding")
        copy.setFixedSize(max(64, copy.fontMetrics().horizontalAdvance(copy.text()) + 20), 38)
        copy.clicked.connect(lambda: clone_command(self.i, controller.randomize_uuid))
        remove = QPushButton("Remove", group)
        remove.setObjectName("removeAction")
        remove.setAccessibleName("Remove embedding")
        remove.setFixedSize(max(76, remove.fontMetrics().horizontalAdvance(remove.text()) + 20), 38)
        remove.clicked.connect(lambda: remove_command(self.i))
        actions.addWidget(copy)
        actions.addWidget(remove)
        group_layout.addLayout(actions)

        self._field_grids: list[tuple[QGridLayout, list[QWidget]]] = []
        ui = self.ui_state
        self._add_fields(group_layout, "Source and tokens", (
            ("base embedding:", "The base embedding to train on. Leave empty to create a new embedding",
             lambda card: pyside6_components.path_entry(card, 1, 0, ui, "model_name", mode="file", path_modifier=path_util.json_path_modifier)),
            ("placeholder:", "The placeholder used when using the embedding in a prompt",
             lambda card: pyside6_components.entry(card, 1, 0, ui, "placeholder")),
            ("token count:", "The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.",
             lambda card: pyside6_components.entry(card, 1, 0, ui, "token_count", width=40)),
        ))
        self._add_fields(group_layout, "Training", (
            ("train:", "", lambda card: pyside6_components.switch(card, 1, 0, ui, "train", command=save_command, width=40)),
            ("output embedding:", "Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.",
             lambda card: pyside6_components.switch(card, 1, 0, ui, "is_output_embedding", width=40)),
            ("stop training after:", "When to stop training the embedding",
             lambda card: pyside6_components.time_entry(card, 1, 0, ui, "stop_training_after", "stop_training_after_unit")),
            ("initial embedding text:", "The initial embedding text used when creating a new embedding",
             lambda card: pyside6_components.entry(card, 1, 0, ui, "initial_embedding_text")),
        ))

        self._column_count = 0
        self._reflow()

    def _add_fields(self, parent_layout: QVBoxLayout, heading: str, fields):
        title = QLabel(heading, self)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        parent_layout.addWidget(title)

        field_grid = QGridLayout()
        field_grid.setHorizontalSpacing(16)
        field_grid.setVerticalSpacing(8)
        cards = []
        for label_text, tooltip, make_control in fields:
            card = QFrame(self)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            card_layout = pyside6_components._layout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setVerticalSpacing(4)
            card_layout.setColumnStretch(0, 1)
            label = pyside6_components.label(card, 0, 0, label_text, tooltip=tooltip)
            label.setWordWrap(True)
            make_control(card)
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
