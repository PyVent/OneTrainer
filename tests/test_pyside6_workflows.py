"""Run Qt workflows in a disposable workspace without model or GPU work."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton

from modules.ui.TopBarController import TopBarController
from modules.util.config.TrainConfig import TrainConfig


class _DeferredThread:
    def __init__(self, target):
        self.target = target
        self.started = False
        self.finished = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.finished

    def run(self):
        try:
            self.target()
        finally:
            self.finished = True


class ConfigDirectoryTest(unittest.TestCase):
    def test_last_session_save_creates_missing_presets_directory(self):
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                controller = TopBarController(TrainConfig.default_values())
                path = controller.save_to_file("#")
                self.assertTrue(Path(path).is_file())
            finally:
                os.chdir(previous_cwd)


class QtWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous_cwd = os.getcwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        for dirname in ("training_presets", "training_configs", "training_concepts", "training_samples"):
            Path(dirname).mkdir()
        (Path("training_concepts") / "concepts.json").write_text("[]", encoding="utf-8")
        (Path("training_samples") / "samples.json").write_text("[]", encoding="utf-8")
        from modules.ui.PySide6TrainUIView import PySide6TrainView
        self.view = PySide6TrainView()
        self.view.show()
        self.app.processEvents()

    def tearDown(self):
        try:
            # A failed assertion must not leave a fake running thread that
            # triggers the modal close confirmation during cleanup.
            self.view.controller.training_thread = None
            self.view.close()
            self.app.processEvents()
        finally:
            os.chdir(self.previous_cwd)
            self.temp.cleanup()

    def test_config_buttons_round_trip_and_export_omit_secrets(self):
        state = self.view.ui_state
        topbar = self.view.top_bar_component
        path = str(Path(self.temp.name) / "training_configs" / "synthetic.json")
        export_path = str(Path(self.temp.name) / "export.json")
        state.get_var("learning_rate").set("0.0002")
        self.view.controller.train_config.secrets.huggingface_token = "synthetic-private-token"

        with patch.object(QFileDialog, "getSaveFileName", return_value=(path, "JSON (*.json)")):
            topbar._save_button.click()
        saved = Path(path).read_text(encoding="utf-8")
        self.assertNotIn("synthetic-private-token", saved)

        state.get_var("learning_rate").set("0.0003")
        with patch.object(QFileDialog, "getOpenFileName", return_value=(path, "JSON (*.json)")):
            topbar._load_button.click()
        self.app.processEvents()
        self.assertEqual(self.view.controller.train_config.learning_rate, 0.0002)

        self.view.controller.train_config.secrets.huggingface_token = "synthetic-private-token"
        with patch.object(QFileDialog, "getSaveFileName", return_value=(export_path, "JSON Files (*.json)")):
            self.view.export_button.click()
        exported = Path(export_path).read_text(encoding="utf-8")
        self.assertNotIn("synthetic-private-token", exported)
        self.assertEqual(json.loads(exported)["learning_rate"], 0.0002)

    def test_training_buttons_delegate_sample_backup_save_and_stop(self):
        trainer = Mock()
        commands = Mock()
        with (
            patch("modules.ui.TrainUIController.flush_and_validate_all", return_value=[]),
            patch("modules.ui.TrainUIController.TrainCommands", return_value=commands),
            patch("modules.ui.TrainUIController.threading.Thread", _DeferredThread),
            patch("modules.ui.TrainUIController.create.create_trainer", return_value=trainer),
            patch("modules.ui.TrainUIController.torch_gc"),
            patch("modules.ui.TrainUIController.torch.clear_autocast_cache"),
            patch.object(self.view, "confirm", return_value=True),
            patch.object(self.view, "show_validation_errors", side_effect=AssertionError("Training preparation failed")),
        ):
            self.view.training_button.click()
            worker = self.view.controller.training_thread
            self.assertIsInstance(worker, _DeferredThread)
            self.assertEqual(self.view.training_button.text(), "Stop Training")

            self.view.tabview.setCurrentWidget(self.view._tab_widgets["sampling"])
            next(button for button in self.view._tab_widgets["sampling"].findChildren(QPushButton)
                 if button.text() == "Sample Now").click()
            self.view.tabview.setCurrentWidget(self.view._tab_widgets["backup"])
            for label in ("Backup Now", "Save Now"):
                next(button for button in self.view._tab_widgets["backup"].findChildren(QPushButton)
                     if button.text() == label).click()
            commands.sample_default.assert_called_once_with()
            commands.backup.assert_called_once_with()
            commands.save.assert_called_once_with()

            self.view.training_button.click()
            commands.stop.assert_called_once_with()
            self.assertEqual(self.view.training_button.text(), "Stopping...")
            worker.run()
            self.app.processEvents()

        trainer.start.assert_called_once_with()
        trainer.train.assert_called_once_with()
        trainer.end.assert_called_once_with()
        self.assertIsNone(self.view.controller.training_thread)
        self.assertEqual(self.view.training_button.text(), "Start Training")

    def test_sampling_tool_reuses_window_and_releases_callbacks_on_close(self):
        class SampleDialog(QDialog):
            def __init__(self, parent, controller):
                super().__init__(parent)
                self.controller = controller

        controller = self.view.controller
        with (
            patch("modules.ui.TrainUIController.SampleWindowController", return_value=object()),
            patch("modules.ui.TrainUIController.torch_gc") as cleanup,
        ):
            controller.open_sampling_tool(self.view, SampleDialog)
            first = controller.sample_window
            self.assertIsInstance(first, SampleDialog)
            self.assertTrue(first.isVisible())
            controller.open_sampling_tool(self.view, SampleDialog)
            self.assertIs(controller.sample_window, first)
            first.accept()
            self.app.processEvents()
            self.assertIsNone(controller.sample_window)
            cleanup.assert_called_once_with()

            callbacks = Mock()
            controller.training_callbacks = callbacks
            controller.training_commands = Mock()
            controller.open_manual_sample_window(self.view, SampleDialog)
            manual = controller.sample_window
            self.assertIsInstance(manual, SampleDialog)
            manual.accept()
            self.app.processEvents()
            callbacks.set_on_sample_custom.assert_called_once_with()
            self.assertIsNone(controller.sample_window)


if __name__ == "__main__":
    unittest.main()
