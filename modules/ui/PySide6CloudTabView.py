from modules.ui.BaseCloudTabView import BaseCloudTabView
from modules.ui.CloudTabController import CloudTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QWidget


class PySide6CloudTabView(BaseCloudTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: CloudTabController, ui_state):
        QWidget.__init__(self, master)
        BaseCloudTabView.__init__(self, pyside6_components, controller)

        self.ui_state = ui_state

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        lo = pyside6_components._layout(frame)
        self.frame = frame

        self.build_content(frame, controller, ui_state)
        self._section_groups = self._group_fields(lo)
        self._section_columns = 0
        self._arrange_sections()

    def _group_fields(self, layout):
        sections = (
            ("Connection", tuple((row, 0) for row in range(10))),
            ("Remote installation", tuple((row, 2) for row in range(1, 7))),
            ("Create a cloud instance", tuple((row, 4) for row in range(1, 7))),
            ("Training connection", ((10, 0), (8, 2), (9, 2))),
            ("Download and cleanup", tuple((row, 2) for row in range(11, 17))),
            ("Instance lifecycle", tuple((row, 4) for row in range(8, 12))),
        )
        fields = {}
        for _, positions in sections:
            for row, col in positions:
                for offset in (0, 1):
                    item = layout.itemAtPosition(row, col + offset)
                    if item is None or item.widget() is None:
                        raise RuntimeError(f"Missing Cloud field at row {row}, column {col + offset}")
                    fields[row, col + offset] = item.widget()
        if layout.count() != len(fields):
            raise RuntimeError("Unexpected widgets in Cloud form")
        while layout.count():
            layout.takeAt(0)

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(14)
        groups = []
        for title, positions in sections:
            group = QGroupBox(title, self.frame)
            form = QFormLayout(group)
            form.setContentsMargins(16, 18, 16, 16)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(10)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            for row, col in positions:
                form.addRow(fields[row, col], fields[row, col + 1])
            groups.append(group)
        return groups

    def _arrange_sections(self):
        columns = 2 if self.width() >= 1040 else 1
        if columns == self._section_columns:
            return
        layout = pyside6_components._layout(self.frame)
        for group in self._section_groups:
            layout.removeWidget(group)
        for row in range(layout.rowCount() + 1):
            layout.setRowStretch(row, 0)
        for index, group in enumerate(self._section_groups):
            layout.addWidget(group, index // columns, index % columns)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1 if columns == 2 else 0)
        layout.setRowStretch((len(self._section_groups) + columns - 1) // columns, 1)
        self._section_columns = columns

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_section_groups"):
            self._arrange_sections()

    def _on_set_gpu_types(self):
        self.gpu_types_menu.clear()
        self.gpu_types_menu.addItems(self.controller.get_gpu_types())

    def _make_reattach_frame(self, frame):
        reattach_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(reattach_frame, 9, 3)
        pyside6_components._layout(reattach_frame).setColumnStretch(0, 1)
        return reattach_frame

    def _make_create_frame(self, frame):
        create_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(create_frame, 1, 5)
        pyside6_components._layout(create_frame).setColumnStretch(1, 1)
        return create_frame
