import os
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit, QPushButton, QWidget

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import pyside6_components as components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.QtVar import QtVar
from modules.util.ui.pyside6_theme import apply_theme


class _ChoiceState:
    def __init__(self, value):
        self.var = QtVar(value)

    def get_var(self, _name):
        return self.var


class ComponentSizingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app, "dark")

    @classmethod
    def tearDownClass(cls):
        apply_theme(cls.app, "light")

    def test_bound_fields_and_browse_button_share_height(self):
        config = TrainConfig.default_values()
        state = PySide6UIState(config)
        frame = QWidget()
        self.addCleanup(frame.close)

        line = components.entry(frame, 0, 0, state, "learning_rate")
        choice = components.options(frame, 1, 0, ["CONSTANT", "CUSTOM"], state, "learning_rate_scheduler")
        timed = components.time_entry(frame, 2, 0, state, "sample_after", "sample_after_unit")
        advanced_action = Mock()
        advanced, controls = components.options_adv(
            frame, 3, 0, ["UNIFORM", "SIGMOID"], state, "timestep_distribution",
            adv_command=advanced_action,
        )
        changed = Mock()
        path = components.path_entry(frame, 4, 0, state, "workspace_dir", mode="dir", command=changed)
        browse = path.findChild(QPushButton)
        path_line = path.findChild(QLineEdit)
        frame.show()
        self.app.processEvents()

        for widget in (line, choice, path_line, browse,
                       controls["component"], controls["button_component"],
                       timed.findChild(QLineEdit), timed.findChild(components.NoScrollComboBox)):
            self.assertEqual(widget.height(), components.CONTROL_HEIGHT)
        self.assertEqual(advanced.height(), components.CONTROL_HEIGHT)
        self.assertEqual(path.height(), components.CONTROL_HEIGHT)
        self.assertGreaterEqual(browse.width(), 88)
        self.assertEqual(browse.text(), "Browse")
        self.assertEqual(controls["button_component"].text(), "Edit")
        self.assertEqual(controls["button_component"].accessibleName(), "Advanced settings")
        controls["button_component"].click()
        advanced_action.assert_called_once_with()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(QFileDialog, "getExistingDirectory", return_value=directory):
                browse.click()
            self.assertEqual(config.workspace_dir, directory)
            changed.assert_called_once_with(directory)

    def test_action_buttons_keep_visible_labels_and_actions(self):
        frame = QWidget()
        self.addCleanup(frame.close)
        remove = Mock()
        duplicate = Mock()
        edit = Mock()
        red = components.colored_icon_button(frame, 0, 0, "X", "#C00000", remove)
        green = components.colored_icon_button(frame, 0, 1, "+", "#00C000", duplicate)
        advanced = components.icon_button(frame, 0, 2, "...", edit)
        frame.show()
        self.app.processEvents()

        for button in (red, green, advanced):
            self.assertEqual(button.height(), components.CONTROL_HEIGHT)
        self.assertEqual(red.accessibleName(), "Remove")
        self.assertEqual(green.accessibleName(), "Duplicate")
        self.assertFalse(red.icon().isNull())
        self.assertFalse(green.icon().isNull())
        self.assertEqual((advanced.text(), advanced.accessibleName()), ("Edit", "Edit settings"))
        self.assertEqual(red.width(), components.CONTROL_HEIGHT)
        self.assertEqual(green.width(), components.CONTROL_HEIGHT)
        red.click()
        green.click()
        advanced.click()
        remove.assert_called_once_with()
        duplicate.assert_called_once_with()
        edit.assert_called_once_with()

    def test_preset_menu_has_no_visible_placeholder_button(self):
        frame = QWidget()
        self.addCleanup(frame.close)
        preset = components.preset_menu_button(
            frame, 0, 0, "Load Preset", [("Default", "default")], Mock()
        )
        frame.show()
        self.app.processEvents()
        self.assertTrue(preset.isVisible())
        self.assertEqual(
            [button for button in frame.findChildren(QPushButton) if button.isVisible()],
            [],
        )

    def test_long_popup_is_bounded_and_selection_stays_bound(self):
        frame = QWidget()
        self.addCleanup(frame.close)
        values = ["short"] + [f"long {index} " + "word " * 100 for index in range(80)]
        state = _ChoiceState(values[60])
        changed = Mock()
        combo = components.options(frame, 0, 0, values, state, "choice", command=changed)
        frame.resize(320, 100)
        frame.show()
        self.app.processEvents()
        self.assertLessEqual(frame.width(), 320)
        self.assertLessEqual(combo.width(), 320)

        combo.showPopup()
        self.app.processEvents()
        popup = combo.view().window()
        self.assertLessEqual(popup.width(), components.COMBO_POPUP_MAX_WIDTH)
        self.assertGreater(popup.width(), combo.width())
        self.assertLessEqual(popup.height(), 320)
        self.assertGreater(combo.view().verticalScrollBar().maximum(), 0)
        self.assertGreaterEqual(
            combo.view().viewport().height(),
            components.COMBO_POPUP_MAX_ROWS * combo.view().sizeHintForRow(0),
        )
        self.assertGreaterEqual(
            popup.y(), combo.mapToGlobal(QPoint(0, combo.height())).y()
        )
        selected = combo.view().visualRect(combo.model().index(combo.currentIndex(), 0))
        self.assertTrue(selected.intersects(combo.view().viewport().rect()))
        combo.hidePopup()

        combo.setCurrentIndex(1)
        self.assertEqual(state.var.get(), values[1])
        changed.assert_called_once_with(values[1])

        kv_state = _ChoiceState("first")
        selected = Mock()
        kv = components.options_kv(frame, 1, 0, [("First", "first"), ("Second", "second")],
                                   kv_state, "choice", command=selected)
        kv.setCurrentText("Second")
        self.assertEqual(kv_state.var.get(), "second")
        self.assertEqual(selected.call_args.args, ("second",))


if __name__ == "__main__":
    unittest.main()
