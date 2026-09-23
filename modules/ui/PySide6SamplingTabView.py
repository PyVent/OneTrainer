from modules.ui.BaseSamplingTabView import BaseSampleWidgetView, BaseSamplingTabView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.ui.PySide6SampleParamsWindowView import PySide6SampleParamsWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QSizePolicy, QWidget


class PySide6SamplingTabView(PySide6ConfigListView, BaseSamplingTabView):

    def __init__(self, master, controller: SamplingTabController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            from_external_file=True,
            attr_name="sample_definition_file_name",
            config_dir="training_samples",
            default_config_name="samples.json",
            add_button_text="Add Sample",
            add_button_tooltip="Add a new sample configuration.",
            is_full_width=True,
            show_toggle_button=True,
        )

    def open_element_window(self, i, ui_state):
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state, PySide6SampleParamsWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6SampleWidgetView(master, element, i, open_command, remove_command, clone_command, save_command)


class PySide6SampleWidgetView(BaseSampleWidgetView, QWidget, metaclass=QtABCMeta):
    WIDE_WIDTH = 1200
    MEDIUM_WIDTH = 720

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        QWidget.__init__(self, master)
        BaseSampleWidgetView.__init__(self, pyside6_components)

        from modules.util.ui.PySide6UIState import PySide6UIState
        self.element = element
        self.ui_state = PySide6UIState(element)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.build_content(self, element, self.ui_state, i, open_command, remove_command, clone_command, save_command)
        layout = pyside6_components._layout(self)
        self._fields = tuple(layout.itemAtPosition(0, column).widget() for column in range(12))
        self._layout_mode = None
        self._reflow_fields(0)
        # Below the compact form's natural minimum the list should scroll,
        # rather than squeezing controls until they overlap or disappear.
        self.setMinimumWidth(self.minimumSizeHint().width())
        self._reflow_fields(self.width())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_fields"):
            self._reflow_fields(event.size().width())

    def _reflow_fields(self, width: int):
        mode = "wide" if width >= self.WIDE_WIDTH else "medium" if width >= self.MEDIUM_WIDTH else "compact"
        if mode == self._layout_mode:
            return

        if mode == "wide":
            positions = [(0, column, 1, 1) for column in range(12)]
            stretch_column = 10
        elif mode == "medium":
            positions = (
                [(0, column, 1, 1) for column in range(9)]
                + [(1, 0, 1, 1), (1, 1, 1, 8), (1, 9, 1, 1)]
            )
            stretch_column = 1
        else:
            # Actions stay on the first row. Size fields and the prompt each
            # get their own space instead of forcing a horizontal scrollbar.
            positions = [
                (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1),
                (1, 0, 1, 1), (1, 1, 1, 1),
                (1, 2, 1, 1), (1, 3, 1, 1),
                (2, 0, 1, 1), (2, 1, 1, 3),
                (3, 0, 1, 1), (3, 1, 1, 3),
                (0, 3, 1, 1),
            ]
            stretch_column = 1

        layout = pyside6_components._layout(self)
        for field in self._fields:
            layout.removeWidget(field)
        for column in range(12):
            layout.setColumnStretch(column, 1 if column == stretch_column else 0)
        for field, (row, column, row_span, column_span) in zip(self._fields, positions):
            layout.addWidget(field, row, column, row_span, column_span)
        self._layout_mode = mode

    def _bind_save(self, save_command):
        self.width_entry.editingFinished.connect(save_command)
        self.height_entry.editingFinished.connect(save_command)
        self.seed_entry.editingFinished.connect(save_command)
        self.prompt_entry.editingFinished.connect(save_command)

    def _set_enabled(self):
        enabled = self.element.enabled
        self.width_entry.setEnabled(enabled)
        self.height_entry.setEnabled(enabled)
        self.prompt_entry.setEnabled(enabled)
        self.seed_entry.setEnabled(enabled)
        self.button.setEnabled(enabled)

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()
