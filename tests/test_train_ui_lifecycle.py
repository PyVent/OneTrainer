"""Exercise the training controller without importing model or GPU dependencies."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_controller():
    stubs = {}

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        if name in {"scripts", "modules", "modules.ui", "modules.util",
                    "modules.util.callbacks", "modules.util.commands",
                    "modules.util.config", "modules.util.ui"}:
            result.__path__ = []
        stubs[name] = result
        if "." in name:
            parent, child = name.rsplit(".", 1)
            if parent in stubs:
                setattr(stubs[parent], child, result)
        return result

    module("scripts")
    module("scripts.generate_debug_report")
    module("modules")
    module("modules.ui")
    for name, symbol in (
        ("BaseTrainUIView", "BaseTrainUIView"),
        ("CaptionUIController", "CaptionUIController"),
        ("ConvertModelUIController", "ConvertModelUIController"),
        ("SampleWindowController", "SampleWindowController"),
        ("VideoToolUIController", "VideoToolUIController"),
    ):
        module(f"modules.ui.{name}", **{symbol: type(symbol, (), {})})
    module("modules.util")
    create = module("modules.util.create")
    module("modules.util.callbacks")
    module("modules.util.callbacks.TrainCallbacks", TrainCallbacks=Mock)
    module("modules.util.commands")
    module("modules.util.commands.TrainCommands", TrainCommands=Mock)
    module("modules.util.config")
    module("modules.util.config.TrainConfig", TrainConfig=type("TrainConfig", (), {}))
    module("modules.util.profiling_util", PeakMemoryRecorder=Mock)
    torch_gc = Mock()
    module("modules.util.torch_util", torch_gc=torch_gc)
    module("modules.util.tensorboard_util", tensorboard_executable=Mock(return_value="tensorboard"))
    module("modules.util.TrainProgress", TrainProgress=type("TrainProgress", (), {}))
    module("modules.util.ui")
    module("modules.util.ui.validation", flush_and_validate_all=Mock(return_value=[]))
    clear_autocast_cache = Mock()
    module("torch", clear_autocast_cache=clear_autocast_cache)

    path = Path(__file__).resolve().parents[1] / "modules/ui/TrainUIController.py"
    spec = importlib.util.spec_from_file_location("_train_ui_lifecycle_subject", path)
    subject = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(subject)
    return subject.TrainUIController, create, torch_gc, clear_autocast_cache


class TrainUILifecycleTest(unittest.TestCase):
    def setUp(self):
        self.Controller, self.create, self.torch_gc, self.clear_autocast_cache = _load_controller()
        self.view = Mock()
        self.view.schedule_on_main_thread.side_effect = lambda callback: callback()
        self.view.get_cloud_reattach.return_value = True
        self.controller = object.__new__(self.Controller)
        self.controller.train_config = SimpleNamespace(
            cloud=SimpleNamespace(enabled=True), tensorboard_always_on=True,
        )
        self.controller.view = self.view
        self.controller.training_thread = object()
        self.controller.training_callbacks = None
        self.controller.training_commands = object()
        self.controller.always_on_tensorboard_subprocess = None
        self.controller._tensorboard_open_thread = None
        self.controller.start_time = 1
        self.controller.start_total_steps = 2
        self.controller._start_always_on_tensorboard = Mock()

    def _assert_cleaned(self, failed):
        self.assertIsNone(self.controller.training_thread)
        self.assertIsNone(self.controller.training_callbacks)
        self.assertIsNone(self.controller.training_commands)
        self.assertIsNone(self.controller.start_time)
        self.assertIsNone(self.controller.start_total_steps)
        self.clear_autocast_cache.assert_called_once_with()
        self.torch_gc.assert_called_once_with()
        self.view.on_training_stopped.assert_called_once_with(failed)
        self.view.on_update_status.assert_called_once_with(
            "Error: check the console for details" if failed else "Stopped"
        )
        self.controller._start_always_on_tensorboard.assert_called_once_with()

    def test_trainer_creation_failure_restores_ui(self):
        self.create.create_trainer = Mock(side_effect=RuntimeError("creation failed"))
        with patch("traceback.print_exc"):
            self.controller._TrainUIController__training_thread_function_impl()
        self._assert_cleaned(True)
        self.view.sync_cloud_secrets.assert_called_once_with()

    def test_start_and_train_failures_still_end_trainer(self):
        for failing_method in ("start", "train"):
            with self.subTest(failing_method=failing_method):
                self.setUp()
                trainer = Mock()
                getattr(trainer, failing_method).side_effect = RuntimeError(failing_method)
                self.create.create_trainer = Mock(return_value=trainer)
                with patch("traceback.print_exc"):
                    self.controller._TrainUIController__training_thread_function_impl()
                self._assert_cleaned(True)
                trainer.end.assert_called_once_with()
                self.assertEqual(self.view.sync_cloud_secrets.call_count, 1 if failing_method == "start" else 2)

    def test_end_failure_is_reported_and_restores_ui(self):
        trainer = Mock()
        trainer.end.side_effect = RuntimeError("end failed")
        self.create.create_trainer = Mock(return_value=trainer)
        with patch("traceback.print_exc"):
            self.controller._TrainUIController__training_thread_function_impl()
        self._assert_cleaned(True)
        trainer.start.assert_called_once_with()
        trainer.train.assert_called_once_with()
        trainer.end.assert_called_once_with()

    def test_success_restores_ui(self):
        trainer = Mock()
        self.create.create_trainer = Mock(return_value=trainer)
        self.controller._TrainUIController__training_thread_function_impl()
        self._assert_cleaned(False)
        trainer.end.assert_called_once_with()
        self.view.sync_cloud_secrets.assert_called_once_with()

    def test_prepare_failure_does_not_enter_running_state(self):
        self.controller.training_thread = None
        self.view.save_default.side_effect = RuntimeError("disk full")
        with patch("traceback.print_exc"):
            self.controller.start_training()
        self.view.on_training_started.assert_not_called()
        self.view.show_validation_errors.assert_called_once_with(
            ["Could not prepare training: disk full"]
        )
        self.assertIsNone(self.controller.training_thread)

    def test_setup_failure_restores_start_button(self):
        self.controller.training_thread = None
        self.controller.train_config.clear_cache_before_training = False
        self.controller.train_config.latent_caching = False
        self.controller.train_config.tensorboard = False
        self.torch_gc.side_effect = RuntimeError("GPU cleanup failed")
        with patch("traceback.print_exc"):
            self.controller.start_training()
        self.view.on_training_started.assert_called_once_with()
        self.view.on_training_stopped.assert_called_once_with(True)
        self.view.on_update_status.assert_called_once_with("Error: training did not start")
        self.view.show_validation_errors.assert_called_once_with(
            ["Could not start training: GPU cleanup failed"]
        )
        self.assertIsNone(self.controller.training_thread)
        self.assertIsNone(self.controller.training_commands)

    def test_tensorboard_button_starts_server_and_opens_when_ready(self):
        self.controller.train_config.tensorboard_port = 6006
        self.controller._tensorboard_is_reachable = Mock(side_effect=[False, False, True])
        process = Mock()
        process.poll.return_value = None
        self.controller._start_always_on_tensorboard.side_effect = (
            lambda: setattr(self.controller, "always_on_tensorboard_subprocess", process)
        )
        browser = Mock()
        with patch.dict(self.Controller.open_tensorboard.__globals__, {
            "webbrowser": SimpleNamespace(open=browser),
        }):
            self.controller.open_tensorboard()
            self.controller._tensorboard_open_thread.join(timeout=3)
        self.controller._start_always_on_tensorboard.assert_called_once_with()
        browser.assert_called_once_with("http://localhost:6006", new=0, autoraise=False)

    def test_tensorboard_launch_error_is_reported(self):
        self.controller._start_always_on_tensorboard = self.Controller._start_always_on_tensorboard.__get__(
            self.controller, self.Controller,
        )
        with tempfile.TemporaryDirectory() as directory:
            self.controller.train_config.workspace_dir = directory
            self.controller.train_config.tensorboard_port = 6006
            self.controller.train_config.tensorboard_expose = False
            with patch.dict(self.Controller._start_always_on_tensorboard.__globals__, {
                    "tensorboard_executable": Mock(side_effect=FileNotFoundError("missing tensorboard.exe"))
            }), \
                    patch("traceback.print_exc"):
                self.controller._start_always_on_tensorboard()
        self.assertIsNone(self.controller.always_on_tensorboard_subprocess)
        self.view.on_update_status.assert_called_once_with(
            "Could not start TensorBoard: missing tensorboard.exe"
        )


if __name__ == "__main__":
    unittest.main()
