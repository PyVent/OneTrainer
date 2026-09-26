import contextlib
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.ui.TopBarController import TopBarController
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


class _SyntheticController(ConvertModelUIController):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = True
        self.worker_thread_ids = []

    def perform_conversion(self):
        self.worker_thread_ids.append(threading.get_ident())
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("Synthetic worker was not released")
        if self.fail:
            raise ValueError("Synthetic conversion error")


class ConvertAsyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_conversion_and_training_offer_the_same_loadable_models(self):
        controller = ConvertModelUIController()
        controller.convert_model_args.training_method = TrainingMethod.FINE_TUNE
        window = controller.create_window(None, PySide6ConvertModelUIView)
        self.addCleanup(window.close)
        combo = window._layout.itemAtPosition(0, 1).widget()
        models = [model for _, model in TopBarController(TrainConfig.default_values()).get_model_types()]
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], [str(model) for model in models])
        for index, model_type in enumerate(models):
            with self.subTest(model=model_type):
                combo.setCurrentIndex(index)
                self.app.processEvents()
                self.assertEqual(controller.convert_model_args.model_type, model_type)
                self.assertTrue(controller.get_output_formats())
                self.assertIsNotNone(create.create_model_loader(model_type, TrainingMethod.FINE_TUNE))
                self.assertIsNotNone(create.create_model_saver(model_type, TrainingMethod.FINE_TUNE))

    def _wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_controller_work_cleans_up_and_never_touches_view(self):
        controller = ConvertModelUIController()
        controller.convert_model_args.input_name = "input.safetensors"
        controller.convert_model_args.output_model_destination = "output.safetensors"
        controller.convert_model_args.output_model_format = ModelFormat.ORIGINAL_SINGLE_FILE

        class View:
            def set_converting(self, _active):
                raise AssertionError("Worker touched a GUI view")

        controller.view = View()
        with (
            patch("modules.ui.ConvertModelUIController.create.create_model_loader", side_effect=ValueError("synthetic load error")),
            patch("modules.ui.ConvertModelUIController.torch_gc") as cleanup,
            self.assertRaisesRegex(ValueError, "synthetic load error"),
        ):
            controller.perform_conversion()
        cleanup.assert_called_once_with()

    def test_model_changes_remove_unsupported_embedding_mode(self):
        controller = ConvertModelUIController()
        controller.convert_model_args.training_method = TrainingMethod.EMBEDDING
        window = controller.create_window(None, PySide6ConvertModelUIView)
        self.addCleanup(window.close)
        self.assertEqual(controller.convert_model_args.training_method, TrainingMethod.EMBEDDING)
        model_var = window.ui_state.get_var("model_type")
        for model_type in (ModelType.ANIMA, ModelType.ANIMA_QWEN21_VAE, ModelType.QWEN, ModelType.FLUX_2):
            with self.subTest(model=model_type):
                model_var.set(str(model_type))
                self.app.processEvents()
                methods = window._layout.itemAtPosition(1, 1).widget()
                self.assertEqual([methods.itemData(i) for i in range(methods.count())],
                                 [str(TrainingMethod.FINE_TUNE), str(TrainingMethod.LORA)])
                self.assertIn(controller.convert_model_args.training_method,
                              controller.convert_model_args.supported_training_methods())
                self.assertIn(controller.convert_model_args.output_model_format,
                              model_type.supported_output_formats(controller.convert_model_args.training_method))

        window.ui_state.get_var("training_method").set(str(TrainingMethod.LORA))
        model_var.set(str(ModelType.STABLE_DIFFUSION_15))
        self.app.processEvents()
        self.assertEqual(controller.convert_model_args.training_method, TrainingMethod.LORA)
        methods = window._layout.itemAtPosition(1, 1).widget()
        self.assertIn(str(TrainingMethod.EMBEDDING), [methods.itemData(i) for i in range(methods.count())])

    def test_unsupported_conversion_is_rejected_before_loading_or_downloading(self):
        controller = ConvertModelUIController()
        controller.convert_model_args.model_type = ModelType.ANIMA_QWEN21_VAE
        controller.convert_model_args.training_method = TrainingMethod.EMBEDDING
        with (
            patch("modules.ui.ConvertModelUIController.create.create_model_loader") as loader,
            patch("modules.ui.ConvertModelUIController.torch_gc") as cleanup,
            self.assertRaisesRegex(ValueError, "does not support EMBEDDING"),
        ):
            controller.perform_conversion()
        loader.assert_not_called()
        cleanup.assert_called_once_with()

    def test_convert_stays_responsive_and_recovers_after_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            previous_dir = os.getcwd()
            os.chdir(temporary_dir)
            controller = _SyntheticController()
            window = controller.create_window(None, PySide6ConvertModelUIView)
            main_thread_id = threading.get_ident()
            gui_updates = []
            original_set_converting = window.set_converting

            def record_gui_update(active):
                gui_updates.append(threading.get_ident())
                original_set_converting(active)

            window.set_converting = record_gui_update
            try:
                window.show()
                self.app.processEvents()
                with patch("modules.ui.PySide6ConvertModelUIView.traceback.print_exc"):
                    window.button.click()
                    self.assertTrue(controller.started.wait(1))
                    self.assertFalse(window.button.isEnabled())

                    marker = []
                    QTimer.singleShot(0, lambda: marker.append("GUI processed event"))
                    self.app.processEvents()
                    self.assertEqual(marker, ["GUI processed event"])

                    window.close()
                    self.assertTrue(window.isVisible())
                    controller.release.set()
                    self._wait_for(lambda: window._conversion_thread is None)
                    self.assertTrue(window.button.isEnabled())
                    self.assertIn("Synthetic conversion error", window._status_label.text())

                controller.started.clear()
                controller.release.clear()
                controller.fail = False
                window.button.click()
                self.assertTrue(controller.started.wait(1))
                self.assertFalse(window.button.isEnabled())
                controller.release.set()
                self._wait_for(lambda: window._conversion_thread is None)
                self.assertTrue(window.button.isEnabled())
                self.assertEqual(window._status_label.text(), "Model converted")
                self.assertEqual(gui_updates, [main_thread_id] * len(gui_updates))
                self.assertTrue(all(worker_id != main_thread_id for worker_id in controller.worker_thread_ids))
            finally:
                controller.release.set()
                if window._conversion_thread is not None:
                    self._wait_for(lambda: window._conversion_thread is None)
                with contextlib.suppress(RuntimeError):
                    window.close()
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
