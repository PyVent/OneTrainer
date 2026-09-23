from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QFrame, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout


class WorkflowNavigation(QFrame):
    """Navigation for the existing training pages, grouped by workflow."""

    page_selected = Signal(str)

    SECTIONS = (
        ("Prepare", (("general", "Overview"), ("model", "Model"), ("data", "Data"), ("concepts", "Concepts"))),
        ("Train", (("training", "Training"), ("LoRA", "LoRA"), ("embedding", "Embedding"),
                   ("additional embeddings", "Additional embeddings"), ("backup", "Backups"))),
        ("Generate", (("sampling", "Sampling"),)),
        ("Utilities", (("tools", "Tools"), ("cloud", "Cloud"))),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowNavigation")
        self.setFixedWidth(220)
        self.setStyleSheet("""
            QFrame#workflowNavigation { background: #f4f6f8; border-right: 1px solid #dce1e6; }
            QTreeWidget { background: transparent; border: none; outline: none; }
            QTreeWidget::item { min-height: 30px; padding: 2px 8px; border-radius: 5px; }
            QTreeWidget::item:selected { background: #dce9f8; color: #173f6b; }
            QTreeWidget::item:hover:!selected { background: #e8edf2; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(10)

        title = QLabel("WORKFLOW", self)
        title.setStyleSheet("font-size: 11px; font-weight: 700; color: #657487; letter-spacing: 1px;")
        layout.addWidget(title)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(10)
        self.tree.setItemsExpandable(False)
        self.tree.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.tree.setAccessibleName("Training workflow pages")
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.tree)

        self._items: dict[str, QTreeWidgetItem] = {}

    def set_pages(self, available: set[str]):
        """Rebuild entries after a training-method or model-type change."""
        was_blocked = self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._items.clear()
            for heading, pages in self.SECTIONS:
                visible = [(key, label) for key, label in pages if key in available]
                if not visible:
                    continue
                section = QTreeWidgetItem(self.tree, [heading.upper()])
                section.setFlags(Qt.ItemFlag.ItemIsEnabled)
                section.setForeground(0, QBrush(QColor("#657487")))
                font = section.font(0)
                font.setBold(True)
                font.setPointSize(max(font.pointSize() - 1, 8))
                section.setFont(0, font)
                for key, label in visible:
                    item = QTreeWidgetItem(section, [label])
                    item.setData(0, Qt.ItemDataRole.UserRole, key)
                    self._items[key] = item
                section.setExpanded(True)
        finally:
            self.tree.blockSignals(was_blocked)

    def select_page(self, key: str | None):
        item = self._items.get(key) if key else None
        was_blocked = self.tree.blockSignals(True)
        try:
            self.tree.setCurrentItem(item)
        finally:
            self.tree.blockSignals(was_blocked)

    def _on_current_item_changed(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None):
        if current is not None:
            key = current.data(0, Qt.ItemDataRole.UserRole)
            if key is not None:
                self.page_selected.emit(key)
