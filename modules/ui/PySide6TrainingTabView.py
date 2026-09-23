from modules.ui.BaseTrainingTabView import BaseTrainingTabView
from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.ui.PySide6OptimizerParamsWindowView import PySide6OptimizerParamsWindowView
from modules.ui.PySide6SchedulerParamsWindowView import PySide6SchedulerParamsWindowView
from modules.ui.PySide6TimestepDistributionWindowView import PySide6TimestepDistributionWindowView
from modules.ui.SchedulerParamsWindowController import SchedulerParamsWindowController
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class PySide6TrainingTabView(BaseTrainingTabView, QWidget, metaclass=QtABCMeta):
    # The section widgets are created once by BaseTrainingTabView. Only their
    # containing columns move when the viewport changes size.
    THREE_COLUMN_WIDTH = 1200
    TWO_COLUMN_WIDTH = 800

    def __init__(self, master, controller: TrainingTabController, ui_state):
        QWidget.__init__(self, master)
        BaseTrainingTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self._columns = ()
        self._column_layout = None
        self._column_count = 0
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.viewport().removeEventFilter(self)
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        self.scroll_frame = scroll

        frame = QWidget()
        scroll.setWidget(frame)

        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        lo.setRowStretch(3, 1)

        column_0 = QWidget(frame)
        column_0.setMinimumWidth(0)
        column_0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(column_0).setColumnStretch(0, 1)

        column_1 = QWidget(frame)
        column_1.setMinimumWidth(0)
        column_1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(column_1).setColumnStretch(0, 1)

        column_2 = QWidget(frame)
        column_2.setMinimumWidth(0)
        column_2.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(column_2).setColumnStretch(0, 1)

        self.build(column_0, column_1, column_2, self.controller, self.ui_state)
        self._label_advanced_buttons()
        self._group_sections((column_0, column_1, column_2))

        for col_widget in (column_0, column_1, column_2):
            lo = pyside6_components._layout(col_widget)
            lo.setRowStretch(lo.rowCount(), 1)

        self._columns = (column_0, column_1, column_2)
        self._column_layout = pyside6_components._layout(frame)
        self._column_count = 0
        scroll.viewport().installEventFilter(self)
        self._reflow_columns(scroll.viewport().width())

    def _group_sections(self, columns):
        titles = {
            "Optimizer": "Optimization",
            "Attention": "Precision and EMA",
            "Train UNet": "UNet",
            "Train Transformer": "Transformer",
            "Train Prior": "Prior",
            "Include Unconditional Transformer": "Unconditional transformer",
            "Offset Noise Weight": "Noise and timesteps",
            "Masked Training": "Masked training",
            "MSE Strength": "Loss",
            "Layer Filter": "Layer selection",
            "Embeddings Learning Rate": "Embeddings",
        }
        for column in columns:
            layout = pyside6_components._layout(column)
            sections = [layout.itemAt(index).widget() for index in range(layout.count())]
            for section in sections:
                if section is None:
                    continue
                label = next(iter(section.findChildren(QLabel)), None)
                first_field = label.text() if label is not None else ""
                if "Text Encoder" in first_field:
                    title = first_field.replace("Train ", "").replace("Include ", "")
                else:
                    title = titles.get(first_field, first_field or "Settings")

                index = layout.indexOf(section)
                row, col, row_span, col_span = layout.getItemPosition(index)
                layout.removeWidget(section)
                if isinstance(section, QFrame):
                    section.setFrameShape(QFrame.Shape.NoFrame)

                group = QGroupBox(title, column)
                group.setAccessibleName(title)
                group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                group_layout = QVBoxLayout(group)
                group_layout.setContentsMargins(2, 2, 2, 2)
                group_layout.addWidget(section)
                layout.addWidget(group, row, col, row_span, col_span)

    def _label_advanced_buttons(self):
        for label in self.findChildren(QLabel):
            if label.text() not in {"Optimizer", "Learning Rate Scheduler", "Timestep Distribution"}:
                continue
            grid = label.parentWidget().layout()
            index = grid.indexOf(label)
            if index < 0:
                continue
            row, column, _, _ = grid.getItemPosition(index)
            item = grid.itemAtPosition(row, column + 1)
            if item is None or item.widget() is None:
                continue
            button = item.widget().findChild(QPushButton)
            if button is not None:
                button.setText("Settings")
                button.setFixedWidth(max(76, button.fontMetrics().horizontalAdvance("Settings") + 20))
                button.setToolTip(f"Configure {label.text().lower()}")

    def eventFilter(self, watched, event):
        if self.scroll_frame is not None and watched is self.scroll_frame.viewport():
            if event.type() == QEvent.Type.Resize:
                self._reflow_columns(event.size().width())
        return super().eventFilter(watched, event)

    def _reflow_columns(self, width: int):
        count = 3 if width >= self.THREE_COLUMN_WIDTH else 2 if width >= self.TWO_COLUMN_WIDTH else 1
        if count == self._column_count:
            return

        layout = self._column_layout
        for column in self._columns:
            layout.removeWidget(column)

        for index in range(3):
            layout.setColumnStretch(index, 1 if index < count else 0)

        if count == 2:
            # The middle source column is usually the tallest. Let it span
            # both rows so the third column begins directly below the first.
            # Placing three whole columns in a regular 2x2 grid left a large
            # empty rectangle before the third column.
            layout.addWidget(self._columns[0], 0, 0)
            layout.addWidget(self._columns[1], 0, 1, 2, 1)
            layout.addWidget(self._columns[2], 1, 0)
        else:
            for index, column in enumerate(self._columns):
                layout.addWidget(column, index // count, index % count)

        self._column_count = count

    def restore_optimizer_config(self, variable: str):
        self.controller.restore_optimizer_config(self.ui_state)

    def restore_scheduler(self, variable: str):
        if not hasattr(self, 'lr_scheduler_adv_comp'):
            return
        self.lr_scheduler_adv_comp.setEnabled(self.controller.is_custom_scheduler_value(variable))

    def open_optimizer_params(self):
        PySide6OptimizerParamsWindowView(self, OptimizerParamsWindowController(self.controller.config), self.ui_state).exec()

    def open_scheduler_params(self):
        PySide6SchedulerParamsWindowView(self, SchedulerParamsWindowController(self.controller.config), self.ui_state).exec()

    def open_timestep_distribution(self):
        PySide6TimestepDistributionWindowView(self, TimestepDistributionWindowController(self.controller.config), self.ui_state).exec()
