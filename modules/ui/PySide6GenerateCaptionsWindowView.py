import os

from modules.ui.PySide6GenerateMasksWindowView import _BatchGenerationThread
from modules.util.ui.pyside6_i18n import set_localized_text
from modules.util.ui.pyside6_i18n import translate as tr

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PySide6GenerateCaptionsWindowView(QDialog):
    _progress_requested = Signal(int, int)

    def __init__(self, parent, controller, path, parent_include_subdirectories):
        super().__init__(parent)
        self.controller = controller
        self._running = False
        self._worker = None
        self._progress_requested.connect(self._apply_progress)
        self.setWindowTitle("Batch generate captions")
        self.resize(460, 350)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.model = QComboBox(self)
        self.model.addItems(["Blip", "Blip2", "WD14 VIT v2"])
        form.addRow("Model", self.model)

        folder_row = QWidget(self)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(path or "", folder_row)
        browse = QPushButton("Browse", folder_row)
        browse.setMinimumWidth(72)
        browse.setToolTip("Choose image folder")
        browse.clicked.connect(self.browse_for_path)
        folder_layout.addWidget(self.path)
        folder_layout.addWidget(browse)
        form.addRow("Folder", folder_row)

        self.initial_caption = QLineEdit(self)
        self.caption_prefix = QLineEdit(self)
        self.caption_postfix = QLineEdit(self)
        form.addRow("Initial caption", self.initial_caption)
        form.addRow("Caption prefix", self.caption_prefix)
        form.addRow("Caption postfix", self.caption_postfix)

        self.mode = QComboBox(self)
        mode_names = ["Replace all captions", "Create if absent", "Add as new line"]
        for name in mode_names:
            self.mode.addItem(tr(name), name)
        self.mode.setProperty("_i18n_combo_sources", mode_names)
        self.mode.setCurrentIndex(self.mode.findData("Create if absent"))
        form.addRow("Mode", self.mode)

        self.include_subdirectories = QCheckBox("Include subfolders", self)
        self.include_subdirectories.setChecked(parent_include_subdirectories)
        form.addRow("", self.include_subdirectories)

        self.progress_label = QLabel("Progress: 0/0", self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        self.create_button = QPushButton("Create captions", self)
        self.create_button.clicked.connect(self.create_captions)
        layout.addWidget(self.create_button)

    def browse_for_path(self):
        path = QFileDialog.getExistingDirectory(self, tr("Choose image folder"), self.path.text())
        if path:
            self.path.setText(path)

    def set_progress(self, value, max_value):
        self._progress_requested.emit(value, max_value)

    @Slot(int, int)
    def _apply_progress(self, value, max_value):
        self.progress.setRange(0, max(1, max_value))
        self.progress.setValue(value)
        set_localized_text(self.progress_label, "Progress: {value}/{max_value}",
                           value=value, max_value=max_value)

    def create_captions(self):
        if self._running:
            return
        if not os.path.isdir(self.path.text()):
            QMessageBox.warning(self, tr("Invalid folder"), tr("Choose an existing image folder."))
            return
        options = {
            "model_name": self.model.currentText(),
            "path": self.path.text(),
            "initial_caption": self.initial_caption.text(),
            "caption_prefix": self.caption_prefix.text(),
            "caption_postfix": self.caption_postfix.text(),
            "mode_str": self.mode.currentData(),
            "include_subdirectories": self.include_subdirectories.isChecked(),
        }
        self._running = True
        self.create_button.setEnabled(False)
        self._worker = _BatchGenerationThread(self.controller.create_captions, options, self)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(str)
    def _on_failed(self, message):
        QMessageBox.critical(self, tr("Caption generation failed"), tr(message))

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
