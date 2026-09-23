import os

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class _BatchGenerationThread(QThread):
    failed = Signal(str)

    def __init__(self, operation, options, parent):
        super().__init__(parent)
        self.operation = operation
        self.options = options

    def run(self):
        try:
            self.operation(**self.options)
        except Exception as exc:
            self.failed.emit(str(exc))


class PySide6GenerateMasksWindowView(QDialog):
    _progress_requested = Signal(int, int)

    def __init__(self, parent, controller, path, parent_include_subdirectories):
        super().__init__(parent)
        self.controller = controller
        self._running = False
        self._worker = None
        self._progress_requested.connect(self._apply_progress)
        self.setWindowTitle("Batch generate masks")
        self.resize(460, 420)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.model = QComboBox(self)
        self.model.addItems(["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"])
        form.addRow("Model", self.model)

        folder_row = QWidget(self)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(path or "", folder_row)
        browse = QPushButton("…", folder_row)
        browse.setFixedWidth(32)
        browse.clicked.connect(self.browse_for_path)
        folder_layout.addWidget(self.path)
        folder_layout.addWidget(browse)
        form.addRow("Folder", folder_row)

        self.prompt = QLineEdit(self)
        form.addRow("Prompt", self.prompt)
        self.mode = QComboBox(self)
        self.mode.addItems([
            "Replace all masks", "Create if absent", "Add to existing",
            "Subtract from existing", "Blend with existing",
        ])
        self.mode.setCurrentText("Create if absent")
        form.addRow("Mode", self.mode)

        self.threshold = QDoubleSpinBox(self)
        self.threshold.setRange(0, 1)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.3)
        form.addRow("Threshold", self.threshold)
        self.smooth = QSpinBox(self)
        self.smooth.setRange(0, 10000)
        self.smooth.setValue(5)
        form.addRow("Smooth", self.smooth)
        self.expand = QSpinBox(self)
        self.expand.setRange(0, 10000)
        self.expand.setValue(10)
        form.addRow("Expand", self.expand)
        self.alpha = QDoubleSpinBox(self)
        self.alpha.setRange(0, 1)
        self.alpha.setDecimals(3)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(1)
        form.addRow("Alpha", self.alpha)

        self.include_subdirectories = QCheckBox("Include subfolders", self)
        self.include_subdirectories.setChecked(parent_include_subdirectories)
        form.addRow("", self.include_subdirectories)

        self.progress_label = QLabel("Progress: 0/0", self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        self.create_button = QPushButton("Create masks", self)
        self.create_button.clicked.connect(self.create_masks)
        layout.addWidget(self.create_button)

    def browse_for_path(self):
        path = QFileDialog.getExistingDirectory(self, "Choose image folder", self.path.text())
        if path:
            self.path.setText(path)

    def set_progress(self, value, max_value):
        self._progress_requested.emit(value, max_value)

    @Slot(int, int)
    def _apply_progress(self, value, max_value):
        self.progress.setRange(0, max(1, max_value))
        self.progress.setValue(value)
        self.progress_label.setText(f"Progress: {value}/{max_value}")

    def create_masks(self):
        if self._running:
            return
        if not os.path.isdir(self.path.text()):
            QMessageBox.warning(self, "Invalid folder", "Choose an existing image folder.")
            return
        options = dict(
            model_name=self.model.currentText(),
            path=self.path.text(),
            prompt=self.prompt.text(),
            mode_str=self.mode.currentText(),
            alpha_str=str(self.alpha.value()),
            threshold_str=str(self.threshold.value()),
            smooth_str=str(self.smooth.value()),
            expand_str=str(self.expand.value()),
            include_subdirectories=self.include_subdirectories.isChecked(),
        )
        self._running = True
        self.create_button.setEnabled(False)
        self._worker = _BatchGenerationThread(self.controller.create_masks, options, self)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(str)
    def _on_failed(self, message):
        QMessageBox.critical(self, "Mask generation failed", message)

    @Slot()
    def _on_finished(self):
        self._running = False
        self.create_button.setEnabled(True)
        self._worker.deleteLater()
        self._worker = None

    def done(self, result):
        if not self._running:
            super().done(result)

    def closeEvent(self, event):
        if self._running:
            event.ignore()
        else:
            super().closeEvent(event)
