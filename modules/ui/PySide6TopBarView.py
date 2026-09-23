from collections.abc import Callable

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLayout, QMessageBox, QSizePolicy, QWidget


class PySide6TopBarView(BaseTopBarView, QWidget):

    def __init__(
            self,
            master,
            controller: TopBarController,
            ui_state,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        QWidget.__init__(self, master)
        BaseTopBarView.__init__(self, pyside6_components)
        # The form switches to a stacked layout as the window narrows. Do not
        # let the current wide layout prevent Qt from delivering that resize.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.frame = QWidget(self)
        pyside6_components._layout(self).setSizeConstraint(QLayout.SetNoConstraint)
        pyside6_components._layout(self).addWidget(self.frame, 0, 0)
        pyside6_components._layout(self.frame).setSizeConstraint(QLayout.SetNoConstraint)
        pyside6_components._layout(self.frame).setContentsMargins(
            pyside6_components.PAD, pyside6_components.PAD,
            pyside6_components.PAD, pyside6_components.PAD,
        )

        self.build(self.frame, master, controller, ui_state,
                   change_model_type_callback, change_training_method_callback, load_preset_callback)

        layout = pyside6_components._layout(self.frame)
        self._title = layout.itemAtPosition(0, 0).widget()
        self._preset_button = layout.itemAtPosition(0, 1).widget()
        self._load_button = layout.itemAtPosition(0, 2).widget()
        self._save_button = layout.itemAtPosition(0, 3).widget()
        self._save_button.setObjectName("primaryHeaderAction")
        self._save_button.setText("Save Config")
        self._load_button.setText("Load Config")
        self._wiki_button = layout.itemAtPosition(0, 4).widget()
        self._model_combo = layout.itemAtPosition(0, 6).widget()
        for button in (self._preset_button, self._load_button, self._save_button):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._actions_frame = QWidget(self.frame)
        actions = QHBoxLayout(self._actions_frame)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addStretch(1)
        for button in (self._preset_button, self._load_button, self._save_button):
            actions.addWidget(button)
        # Branding and Help live in the navigation sidebar. Keep the built
        # widgets so the shared callbacks remain available, but hide them here.
        self._title.hide()
        self._wiki_button.hide()
        self._model_label = QLabel("Model", self.frame)
        self._method_label = QLabel("Training method", self.frame)
        self._model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._model_combo.setMaximumWidth(420)
        self._layout_mode = None
        self._placed_method = None
        self._reflow()

        # BaseTopBarView replaces the method combo when the model changes.
        # Defer layout until that replacement has finished, including changes
        # originating from a loaded configuration rather than a mouse click.
        self._model_combo.currentTextChanged.connect(
            lambda _text: QTimer.singleShot(0, self._reflow)
        )

    def _setup_frame_column_weight(self):
        # Column stretch is assigned by _reflow after BaseTopBarView builds its widgets.
        pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_model_combo"):
            self._reflow()

    def _reflow(self):
        width = self.width()
        mode = "wide" if width >= 1220 else "medium" if width >= 850 else "compact"
        method = self.training_method
        if mode == self._layout_mode and method is self._placed_method:
            return

        layout = pyside6_components._layout(self.frame)
        widgets = (
            self._title, self._wiki_button, self._model_combo, self._actions_frame,
            self._model_label, self._method_label, self._placed_method, method,
        )
        for widget in widgets:
            if widget is not None:
                layout.removeWidget(widget)
        for column in range(max(layout.columnCount(), 8)):
            layout.setColumnStretch(column, 0)

        def place(widget, row, column, span=1):
            if widget is not None:
                layout.addWidget(widget, row, column, 1, span, Qt.AlignmentFlag(0))

        if mode == "wide":
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1)
            place(self._method_label, 0, 2)
            place(method, 0, 3)
            place(self._actions_frame, 0, 4)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(3, 1)
            layout.setColumnStretch(4, 1)
        elif mode == "medium":
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1)
            place(self._method_label, 0, 2)
            place(method, 0, 3)
            place(self._actions_frame, 1, 0, 4)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(3, 1)
        else:
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1, 2)
            place(self._method_label, 1, 0)
            place(method, 1, 1, 2)
            place(self._actions_frame, 2, 0, 3)
            layout.setColumnStretch(1, 1)

        self._layout_mode = mode
        self._placed_method = method

    def _forget_dropdown(self, widget):
        lo = pyside6_components._layout(self.frame)
        lo.removeWidget(widget)
        if getattr(self, "_placed_method", None) is widget:
            self._placed_method = None
        widget.hide()
        widget.deleteLater()

    def _show_save_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", initial_dir, "JSON (*.json)")
        if path:
            # the native dialog doesn't reliably append the filter's extension on every platform
            if not path.endswith(".json"):
                path += ".json"
            try:
                callback(path)
            except OSError as exc:
                QMessageBox.critical(self, "Cannot save configuration", str(exc))

    def _show_open_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", initial_dir, "JSON (*.json)")
        if path:
            callback(path)

    def _show_load_error(self, message: str):
        QMessageBox.warning(self, "Cannot load configuration", message)
