import traceback

from modules.ui.BaseConvertModelUIView import BaseConvertModelUIView
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.pyside6_i18n import set_localized_text, translate as tr

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QLayout, QWidget


class _ConversionWorker(QObject):
    finished = Signal(bool, str)

    def __init__(self, controller: ConvertModelUIController):
        super().__init__()
        self.controller = controller

    @Slot()
    def run(self):
        try:
            self.controller.perform_conversion()
        except Exception as error:
            traceback.print_exc()
            self.finished.emit(False, f"{type(error).__name__}: {error}")
        else:
            self.finished.emit(True, "")


class PySide6ConvertModelUIView(BaseConvertModelUIView, QDialog):
    def __init__(self, parent, controller: ConvertModelUIController):
        QDialog.__init__(self, parent)
        BaseConvertModelUIView.__init__(self, pyside6_components)

        self.controller = controller
        self.ui_state = PySide6UIState(controller.convert_model_args)
        self._dynamic_frame = None
        self._conversion_thread: QThread | None = None
        self._conversion_worker: _ConversionWorker | None = None
        self._conversion_result: tuple[bool, str] | None = None

        self.setWindowTitle("Convert models")
        self.resize(600, 380)

        _pad = pyside6_components.PAD
        outer = QGridLayout(self)
        outer.setContentsMargins(_pad, _pad, _pad, _pad)
        outer.setRowStretch(0, 1)

        self._scroll_area, self._frame = pyside6_components.scrollable_frame(self)
        self._scroll_area.widget().layout().setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self._layout = pyside6_components._layout(self._frame)
        self._layout.setColumnStretch(1, 1)
        outer.addWidget(self._scroll_area, 0, 0)

        self.build_content(self._frame, controller, self.ui_state, self._rebuild_dynamic_ui)
        self.button.setObjectName("primaryAction")
        training_method_label = self._layout.itemAtPosition(1, 0).widget()
        if isinstance(training_method_label, QLabel):
            training_method_label.setText("Training Method")
        self._rebuild_dynamic_ui()
        self._layout.setRowStretch(self._layout.rowCount(), 1)

        self._status_label = QLabel("", self)
        self._status_label.setObjectName("convertStatus")
        self._status_label.setWordWrap(True)
        self._status_label.setAccessibleName("Conversion status")
        outer.addWidget(self._status_label, 1, 0)

    def _rebuild_dynamic_ui(self, *args):
        if self._dynamic_frame is not None:
            self._dynamic_frame.hide()
            self._dynamic_frame.deleteLater()

        self._dynamic_frame = QWidget(self._frame)
        self._layout.addWidget(self._dynamic_frame, 4, 0, 1, 2)

        self.build_dynamic_content(self._dynamic_frame, self.controller, self.ui_state)

    def set_converting(self, active):
        self.button.setEnabled(not active)
        self._frame.setEnabled(not active)

    def _set_status_error(self, error: bool):
        self._status_label.setObjectName("errorStatus" if error else "convertStatus")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)
        self._status_label.update()

    @Slot()
    def start_conversion(self):
        if self._conversion_thread is not None:
            return

        self._conversion_result = None
        self._set_status_error(False)
        self._status_label.setText(tr("Converting model..."))
        self.set_converting(True)

        thread = QThread(self)
        worker = _ConversionWorker(self.controller)
        worker.moveToThread(thread)
        self._conversion_thread = thread
        self._conversion_worker = worker

        thread.started.connect(worker.run)
        worker.finished.connect(self._record_conversion_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._conversion_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(bool, str)
    def _record_conversion_result(self, success: bool, error: str):
        self._conversion_result = (success, error)

    @Slot()
    def _conversion_thread_finished(self):
        success, error = self._conversion_result or (False, "Conversion stopped unexpectedly")
        if success:
            self._status_label.setText(tr("Model converted"))
        else:
            set_localized_text(self._status_label, "Conversion failed: {error}", error=error)
            self._set_status_error(True)
        self.set_converting(False)
        self._conversion_thread = None
        self._conversion_worker = None

    def _conversion_running(self) -> bool:
        return self._conversion_thread is not None and self._conversion_thread.isRunning()

    def reject(self):
        if self._conversion_running():
            self._status_label.setText(tr("Conversion is still running"))
            return
        super().reject()

    def closeEvent(self, event):
        if self._conversion_running():
            self._status_label.setText(tr("Conversion is still running"))
            event.ignore()
            return
        super().closeEvent(event)
