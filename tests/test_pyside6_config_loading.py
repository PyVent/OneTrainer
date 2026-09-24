import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.ui.PySide6TopBarView import PySide6TopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox


class _ControllerWithoutPresets(TopBarController):
    def load_preset_tree(self, dir="training_presets"):
        return []

    def load_config_from_file(self, filename):
        self.last_load_error = f"File not found: {filename}"


class ConfigLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_invalid_config_keeps_current_values_and_records_error(self):
        config = TrainConfig.default_values()
        controller = TopBarController(config)
        original_model = config.model_type
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "broken.json"
            filename.write_text("{broken", encoding="utf-8")
            with patch("builtins.print"):
                self.assertIsNone(controller.load_config_from_file(str(filename)))
        self.assertEqual(config.model_type, original_model)
        self.assertTrue(controller.last_load_error)

    def test_invalid_field_keeps_current_values_and_names_the_field(self):
        config = TrainConfig.default_values()
        controller = TopBarController(config)
        original_epochs = config.epochs

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "invalid-value.json"
            data = config.to_settings_dict(secrets=False)
            data["epochs"] = "many"
            filename.write_text(json.dumps(data), encoding="utf-8")
            with patch("builtins.print"):
                self.assertIsNone(controller.load_config_from_file(str(filename)))

        self.assertEqual(config.epochs, original_epochs)
        self.assertIn("epochs", controller.last_load_error)

    def test_qt_view_shows_user_selected_file_error(self):
        config = TrainConfig.default_values()
        controller = _ControllerWithoutPresets(config)
        with patch.object(QMessageBox, "warning") as warning:
            view = PySide6TopBarView(
                None, controller, PySide6UIState(config),
                lambda _model: None, lambda _method: None, lambda: None,
            )
            self.addCleanup(view.close)
            warning.assert_not_called()  # missing last-session file is expected
            view._load_current_config("missing-config.json")
            warning.assert_called_once()
            self.assertIn("File not found", warning.call_args.args[2])

    def test_qt_view_reports_save_failure(self):
        config = TrainConfig.default_values()
        view = PySide6TopBarView(
            None, _ControllerWithoutPresets(config), PySide6UIState(config),
            lambda _model: None, lambda _method: None, lambda: None,
        )
        self.addCleanup(view.close)
        save = Mock(side_effect=PermissionError("access denied"))
        with (
            patch.object(QFileDialog, "getSaveFileName", return_value=("output", "JSON (*.json)")),
            patch.object(QMessageBox, "critical") as critical,
        ):
            view._show_save_dialog("training_configs", save)
        save.assert_called_once_with("output.json")
        critical.assert_called_once()
        self.assertIn("access denied", critical.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
