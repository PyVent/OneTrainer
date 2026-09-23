from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget,
)

from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView
from modules.util.ui.pyside6_i18n import translate as tr


class _MaskPreview(QLabel):
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #202020;")

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            # The controller connects brush positions; start each stroke here.
            self.view.controller.mask_draw_x = event.position().x()
            self.view.controller.mask_draw_y = event.position().y()
            self.view.edit_mask(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            self.view.edit_mask(event)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        if self.view.enable_mask_editing.isChecked():
            steps = event.angleDelta().y() / 120
            if steps:
                self.view.draw_mask_radius(steps)
                event.accept()
                return
        super().wheelEvent(event)


class PySide6CaptionUIView(QDialog):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        controller.view = self
        self.setWindowTitle("Dataset Tool")
        self.resize(1280, 980)

        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        outer.addLayout(top)
        for title, callback in (
            ("Open folder", self.open_directory),
            ("Generate masks", self.open_mask_window),
            ("Generate captions", self.open_caption_window),
            ("Open in Explorer", self.open_in_explorer),
        ):
            button = QPushButton(title, self)
            button.clicked.connect(callback)
            top.addWidget(button)
        self.include_subdirectories = QCheckBox("Include subfolders", self)
        self.include_subdirectories.setChecked(controller.config_ui_data["include_subdirectories"])
        self.include_subdirectories.toggled.connect(self._on_subdirectories_toggled)
        top.addWidget(self.include_subdirectories)
        top.addStretch()
        help_button = QPushButton("Help", self)
        help_button.clicked.connect(self.show_help)
        top.addWidget(help_button)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        outer.addWidget(splitter, 1)
        self.file_list = QListWidget(splitter)
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        splitter.addWidget(self.file_list)

        content = QWidget(splitter)
        content_layout = QVBoxLayout(content)
        controls = QHBoxLayout()
        content_layout.addLayout(controls)
        self.draw_button = QPushButton("Draw", self)
        self.fill_button = QPushButton("Fill", self)
        for button, mode in ((self.draw_button, "draw"), (self.fill_button, "fill")):
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, selected=mode: self._set_mask_mode(selected))
            controls.addWidget(button)
        self._set_mask_mode("draw")

        self.enable_mask_editing = QCheckBox("Edit mask", self)
        controls.addWidget(self.enable_mask_editing)
        self.show_only_mask = QCheckBox("Show mask only", self)
        self.show_only_mask.toggled.connect(self._on_show_mask_toggled)
        controls.addWidget(self.show_only_mask)

        controls.addWidget(QLabel("Alpha", self))
        self.mask_alpha = QDoubleSpinBox(self)
        self.mask_alpha.setRange(0, 1)
        self.mask_alpha.setSingleStep(0.1)
        self.mask_alpha.setValue(1)
        controls.addWidget(self.mask_alpha)
        controls.addWidget(QLabel("Brush %", self))
        self.brush_size = QDoubleSpinBox(self)
        self.brush_size.setRange(0.25, 50)
        self.brush_size.setDecimals(2)
        self.brush_size.setSingleStep(0.25)
        self.brush_size.setValue(controller.mask_draw_radius * 100)
        self.brush_size.valueChanged.connect(self._on_brush_size_changed)
        controls.addWidget(self.brush_size)
        controls.addStretch()

        scroll = QScrollArea(content)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidgetResizable(False)
        self.image_label = _MaskPreview(self)
        self.image_label.setFixedSize(controller.image_size, controller.image_size)
        scroll.setWidget(self.image_label)
        content_layout.addWidget(scroll, 1)

        caption_row = QHBoxLayout()
        content_layout.addLayout(caption_row)
        caption_row.addWidget(QLabel("Caption", self))
        self.prompt_component = QLineEdit(self)
        caption_row.addWidget(self.prompt_component, 1)
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save)
        caption_row.addWidget(save_button)
        nav_row = QHBoxLayout()
        content_layout.addLayout(nav_row)
        for title, callback in (("Previous", controller.previous_image), ("Next", controller.next_image)):
            button = QPushButton(title, self)
            button.clicked.connect(callback)
            nav_row.addWidget(button)
        nav_row.addStretch()
        splitter.addWidget(content)
        splitter.setSizes([300, 950])

        for key, callback in (
            ("Up", controller.previous_image),
            ("Down", controller.next_image),
            ("Return", self.save),
            ("Ctrl+M", self.toggle_mask),
            ("Ctrl+D", lambda: self._set_mask_mode("draw")),
            ("Ctrl+F", lambda: self._set_mask_mode("fill")),
        ):
            shortcut = QShortcut(QKeySequence(key), self.prompt_component)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)

        controller.load_directory(controller.config_ui_data["include_subdirectories"])

    def _on_subdirectories_toggled(self, checked):
        self.controller.config_ui_data["include_subdirectories"] = checked
        self.controller.load_directory(checked)

    def _on_row_changed(self, row):
        if row >= 0 and row != self.controller.current_image_index:
            self.controller.switch_image(row)

    def _set_mask_mode(self, mode):
        self.controller.set_mask_editing_mode(mode)
        self.draw_button.setChecked(mode == "draw")
        self.fill_button.setChecked(mode == "fill")

    def _on_brush_size_changed(self, value):
        self.controller.mask_draw_radius = value / 100

    def _on_show_mask_toggled(self, checked):
        if self.controller.display_only_mask != checked:
            self.controller.toggle_mask()
        if self.controller.pil_image is not None:
            self.refresh_image()

    def refresh_file_list(self):
        self.file_list.blockSignals(True)
        try:
            self.file_list.clear()
            self.file_list.addItems(self.controller.image_rel_paths)
        finally:
            self.file_list.blockSignals(False)

    def focus_prompt(self):
        self.prompt_component.setFocus()

    def on_image_switched(self, old_index, new_index, prompt):
        self.file_list.blockSignals(True)
        try:
            self.file_list.setCurrentRow(new_index)
        finally:
            self.file_list.blockSignals(False)
        self.prompt_component.setText(prompt)
        self.refresh_image()

    def on_image_cleared(self):
        self.file_list.blockSignals(True)
        try:
            self.file_list.setCurrentRow(-1)
        finally:
            self.file_list.blockSignals(False)
        self.image_label.clear()
        self.image_label.setText(tr("No images in this folder"))
        self.prompt_component.clear()

    def refresh_image(self):
        image, size = self.controller.get_display_image()
        pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
        self.image_label.setFixedSize(*size)
        self.image_label.setPixmap(pixmap)

    def draw_mask_radius(self, steps):
        self.controller.update_mask_draw_radius(steps)
        self.brush_size.blockSignals(True)
        try:
            self.brush_size.setValue(self.controller.mask_draw_radius * 100)
        finally:
            self.brush_size.blockSignals(False)

    def edit_mask(self, event):
        if not self.enable_mask_editing.isChecked():
            return
        buttons = event.buttons() | event.button()
        is_left = bool(buttons & Qt.MouseButton.LeftButton)
        is_right = bool(buttons & Qt.MouseButton.RightButton)
        if not (is_left or is_right):
            return
        point = event.position()
        if not self.image_label.rect().contains(point.toPoint()):
            return
        self.controller.handle_edit_mask(
            point.x(), point.y(), is_left, is_right, self.mask_alpha.value()
        )

    def save(self):
        self.controller.save(self.prompt_component.text())

    def toggle_mask(self):
        self.show_only_mask.setChecked(not self.show_only_mask.isChecked())

    def open_directory(self):
        path = QFileDialog.getExistingDirectory(self, tr("Choose image folder"), self.controller.dir or "")
        if path:
            self.controller.dir = path
            self.controller.load_directory(self.include_subdirectories.isChecked())

    def open_mask_window(self):
        dialog = self.controller.open_mask_window(self, PySide6GenerateMasksWindowView)
        dialog.exec()
        self._refresh_current_image()

    def open_caption_window(self):
        dialog = self.controller.open_caption_window(self, PySide6GenerateCaptionsWindowView)
        dialog.exec()
        self._refresh_current_image()

    def _refresh_current_image(self):
        if 0 <= self.controller.current_image_index < len(self.controller.image_rel_paths):
            self.controller.switch_image(self.controller.current_image_index)

    def open_in_explorer(self):
        self.controller.open_in_explorer()

    def show_help(self):
        QMessageBox.information(self, tr("Dataset Tool shortcuts"), tr(self.controller.help_text))

    def done(self, result):
        self.controller._release_models()
        super().done(result)
