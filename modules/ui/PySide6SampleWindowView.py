import threading

from modules.modelSampler.BaseModelSampler import (
    ModelSamplerOutput,
)
from modules.ui.BaseSampleWindowView import BaseSampleWindowView
from modules.ui.PySide6SampleFrameView import PySide6SampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.ui.SampleWindowController import SampleWindowController
from modules.util.enum.FileType import FileType
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_i18n import set_localized_text
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.validation import flush_and_validate_for_save

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QProgressBar, QPushButton, QSizePolicy


class PySide6SampleWindowView(BaseSampleWindowView, QDialog):
    def __init__(self, parent, controller: SampleWindowController):
        QDialog.__init__(self, parent)
        BaseSampleWindowView.__init__(self, pyside6_components)

        self.setWindowTitle("Sample")
        self.resize(1200, 800)

        self.ui_state = PySide6UIState(controller.sample)

        if controller.use_external_model:
            controller.callbacks.set_on_sample_custom(self._update_preview)
            controller.callbacks.set_on_update_sample_custom_progress(self._update_progress)

        outer = QGridLayout(self)
        outer.setRowStretch(1, 1)
        outer.setColumnStretch(1, 1)

        model_type = controller.get_model_type()
        frame_controller = SampleFrameController(controller.sample, model_type)

        prompt_frame = PySide6SampleFrameView(self, frame_controller, self.ui_state, include_settings=False)
        outer.addWidget(prompt_frame, 0, 0, 1, 2)

        settings_frame = PySide6SampleFrameView(self, frame_controller, self.ui_state, include_prompt=False)
        outer.addWidget(settings_frame, 1, 0)

        self._image_label = QLabel(self)
        self._preview_pixmap = None
        self._image_label.setMinimumSize(160, 160)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background: black;")
        outer.addWidget(self._image_label, 1, 1, 3, 1)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1000)
        outer.addWidget(self._progress, 2, 0)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        outer.addWidget(self._status, 4, 0, 1, 2)

        sample_btn = QPushButton("Sample", self)
        sample_btn.setObjectName("primaryAction")
        def _on_sample():
            errors = flush_and_validate_for_save(self.ui_state)
            if errors:
                self._status.setText("\n".join(errors))
                return
            # With an external model (manual sample during training) do_sample
            # only enqueues a command for the training thread to execute, so it
            # returns immediately; queueing several is fine, no thread needed.
            if controller.use_external_model:
                controller.do_sample(self._update_preview, self._update_progress)
                return

            # Standalone sampling runs the model synchronously, so run it in a
            # background thread to keep the Qt event loop responsive, and disable
            # the button while a sample is in flight to avoid concurrent runs.
            sample_btn.setEnabled(False)
            set_localized_text(self._status, "Loading model and preparing generation…")
            self._progress.setRange(0, 0)

            def _run():
                try:
                    controller.do_sample(self._update_preview, self._update_progress)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self.schedule_on_main_thread(lambda: set_localized_text(self._status, "Generation failed: {error}", error=message))
                finally:
                    self.schedule_on_main_thread(lambda: sample_btn.setEnabled(True))
                    self.schedule_on_main_thread(lambda: self._progress.setRange(0, 1000))

            threading.Thread(target=_run, daemon=True).start()
        sample_btn.clicked.connect(_on_sample)
        outer.addWidget(sample_btn, 3, 0)


    def schedule_on_main_thread(self, fn):
        QTimer.singleShot(0, self, fn)

    def _update_preview(self, sampler_output: ModelSamplerOutput):
        # Called from training thread — capture data and dispatch to main thread
        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            self.schedule_on_main_thread(lambda: self._do_update_preview(image))

    def _do_update_preview(self, image):
        self._preview_pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
        self._scale_preview()
        set_localized_text(self._status, "Generation complete")

    def _scale_preview(self):
        if self._preview_pixmap is not None:
            self._image_label.setPixmap(self._preview_pixmap.scaled(
                self._image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_preview_pixmap"):
            self._scale_preview()

    def _update_progress(self, progress: int, max_progress: int):
        # Called from training thread — dispatch to main thread
        def update():
            self._progress.setRange(0, 1000)
            self._progress.setValue(int(progress / max(1, max_progress) * 1000))
            set_localized_text(self._status, "Generating: {step}/{total}", step=progress, total=max_progress)
        self.schedule_on_main_thread(update)
