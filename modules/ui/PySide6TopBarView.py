from collections.abc import Callable

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLayout, QMenu,
    QMessageBox, QPushButton, QSizePolicy, QToolButton, QWidget,
)

from modules.ui.TopBarController import TopBarController
from modules.util import path_util
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.ui.pyside6_components import NoScrollComboBox, PAD
from modules.util.ui.pyside6_i18n import translate as tr


class PySide6TopBarView(QWidget):
    def __init__(
        self,
        master,
        controller: TopBarController,
        ui_state,
        change_model_type_callback: Callable[[ModelType], None],
        change_training_method_callback: Callable[[TrainingMethod], None],
        load_preset_callback: Callable[[], None],
    ):
        super().__init__(master)
        self.controller = controller
        self.ui_state = ui_state
        self.change_model_type_callback = change_model_type_callback
        self.change_training_method_callback = change_training_method_callback
        self.load_preset_callback = load_preset_callback
        self.dir = "training_presets"
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout = QGridLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(PAD)

        self._model_label = QLabel("Model", self)
        self._method_label = QLabel("Training method", self)
        self._model_combo = NoScrollComboBox(self)
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._model_combo.setMaximumWidth(420)
        self.training_method = NoScrollComboBox(self)

        self._preset_button = QToolButton(self)
        self._preset_button.setObjectName("presetMenuButton")
        self._preset_button.setText("Load Preset")
        self._preset_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._preset_menu = QMenu(self._preset_button)
        self._preset_submenus = []
        self._add_presets(self._preset_menu, controller.load_preset_tree(self.dir))
        self._preset_button.setMenu(self._preset_menu)

        self._load_button = QPushButton("Load Config", self)
        self._load_button.setToolTip("Load one of your own saved configs")
        self._load_button.clicked.connect(self._load_config)
        self._save_button = QPushButton("Save Config", self)
        self._save_button.setObjectName("primaryHeaderAction")
        self._save_button.setToolTip("Save the current configuration in a custom preset")
        self._save_button.clicked.connect(self._save_config)
        self._preset_button.setFixedHeight(self._load_button.sizeHint().height())

        self._actions_frame = QWidget(self)
        actions = QHBoxLayout(self._actions_frame)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addStretch(1)
        for button in (self._preset_button, self._load_button, self._save_button):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            actions.addWidget(button)

        self._layout_mode = None
        self._model_combo.currentIndexChanged.connect(self._model_selected)
        self.training_method.currentIndexChanged.connect(self._method_selected)
        ui_state.get_var("model_type").subscribe(self._model_value_changed, owner=self._model_combo)
        ui_state.get_var("training_method").subscribe(self._method_value_changed, owner=self.training_method)
        self._fill_combo(self._model_combo, controller.get_model_types())
        self._model_value_changed(ui_state.get_var("model_type").get())
        self._reflow()
        self._load_current_config(path_util.canonical_join(self.dir, "#.json"), quiet=True)

    def _add_presets(self, menu: QMenu, nodes):
        for name, value in nodes:
            if isinstance(value, list):
                # PySide6 6.11 can hand out a dead QMenu via QAction.menu()
                # when using the string overload of addMenu().
                submenu = QMenu(name, menu)
                menu.addMenu(submenu)
                self._preset_submenus.append(submenu)
                self._add_presets(submenu, value)
            else:
                menu.addAction(name, lambda _checked=False, path=value: self._load_current_config(path))

    @staticmethod
    def _fill_combo(combo: NoScrollComboBox, values):
        with QSignalBlocker(combo):
            combo.clear()
            combo.setProperty("_i18n_combo_sources", [label for label, _ in values])
            for label, value in values:
                combo.addItem(tr(label), value)

    @staticmethod
    def _index_for_value(combo: NoScrollComboBox, value) -> int:
        for index in range(combo.count()):
            if str(combo.itemData(index)) == str(value):
                return index
        return -1

    def _model_selected(self, index: int):
        if index >= 0:
            self.ui_state.get_var("model_type").set(str(self._model_combo.itemData(index)))

    def _method_selected(self, index: int):
        if index >= 0:
            self.ui_state.get_var("training_method").set(str(self.training_method.itemData(index)))

    def _model_value_changed(self, value):
        index = self._index_for_value(self._model_combo, value)
        if index < 0:
            return
        with QSignalBlocker(self._model_combo):
            self._model_combo.setCurrentIndex(index)
        model_type = self._model_combo.itemData(index)
        self.change_model_type_callback(model_type)
        self._fill_combo(self.training_method, self.controller.get_training_methods(model_type))
        method_value = self.ui_state.get_var("training_method").get()
        method_index = self._index_for_value(self.training_method, method_value)
        if method_index < 0 and self.training_method.count():
            self.ui_state.get_var("training_method").set(str(self.training_method.itemData(0)))
        else:
            self._method_value_changed(method_value)

    def _method_value_changed(self, value):
        index = self._index_for_value(self.training_method, value)
        if index < 0:
            return
        with QSignalBlocker(self.training_method):
            self.training_method.setCurrentIndex(index)
        self.change_training_method_callback(self.training_method.itemData(index))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def fit_action_labels(self):
        """Give translated action captions their measured width."""
        buttons = (self._preset_button, self._load_button, self._save_button)
        for button in buttons:
            button.setMinimumWidth(button.sizeHint().width() + 4)
            button.updateGeometry()
        self._actions_frame.setMinimumWidth(
            sum(button.minimumWidth() for button in buttons) + 2 * 8
        )
        self._actions_frame.updateGeometry()
        self.layout().invalidate()

    def _reflow(self):
        width = self.width()
        mode = "wide" if width >= 1220 else "medium" if width >= 850 else "compact"
        if mode == self._layout_mode:
            return

        layout = self.layout()
        widgets = (
            self._model_label, self._model_combo, self._method_label,
            self.training_method, self._actions_frame,
        )
        for widget in widgets:
            layout.removeWidget(widget)
        for column in range(max(layout.columnCount(), 5)):
            layout.setColumnStretch(column, 0)

        def place(widget, row, column, span=1):
            layout.addWidget(widget, row, column, 1, span, Qt.AlignmentFlag(0))

        if mode == "wide":
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1)
            place(self._method_label, 0, 2)
            place(self.training_method, 0, 3)
            place(self._actions_frame, 0, 4)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(3, 1)
            layout.setColumnStretch(4, 1)
        elif mode == "medium":
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1)
            place(self._method_label, 0, 2)
            place(self.training_method, 0, 3)
            place(self._actions_frame, 1, 0, 4)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(3, 1)
        else:
            place(self._model_label, 0, 0)
            place(self._model_combo, 0, 1, 2)
            place(self._method_label, 1, 0)
            place(self.training_method, 1, 1, 2)
            place(self._actions_frame, 2, 0, 3)
            layout.setColumnStretch(1, 1)

        self._layout_mode = mode

    def _load_config(self):
        self._show_open_dialog("training_configs", self._load_current_config)

    def _save_config(self):
        self._show_save_dialog("training_configs", self.controller.save_config_to_path)

    def _load_current_config(self, filename, *, quiet: bool = False):
        loaded_config = self.controller.load_config_from_file(filename)
        if loaded_config is None:
            if not quiet:
                self._show_load_error(
                    self.controller.last_load_error
                    or tr("Could not load {filename}").format(filename=filename)
                )
            return

        self.ui_state.update(loaded_config)
        optimizer_config = change_optimizer(self.controller.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)
        self.load_preset_callback()

    def _show_save_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getSaveFileName(self, tr("Save config"), initial_dir, "JSON (*.json)")
        if path:
            if not path.endswith(".json"):
                path += ".json"
            try:
                callback(path)
            except OSError as exc:
                QMessageBox.critical(self, tr("Cannot save configuration"), str(exc))

    def _show_open_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getOpenFileName(self, tr("Load config"), initial_dir, "JSON (*.json)")
        if path:
            callback(path)

    def _show_load_error(self, message: str):
        QMessageBox.warning(self, tr("Cannot load configuration"), tr(message))

    def open_wiki(self):
        self.controller.open_wiki()

    def save_default(self):
        self.controller.save_default()
