import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.util.ui.pyside6_i18n import (
    current_language,
    retranslate_tree,
    saved_language,
    set_language,
    set_localized_text,
    translate,
)
from modules.util.ui.pyside6_navigation import WorkflowNavigation

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class LanguageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_language = current_language()
        set_language(self.app, "en", persist=False)

    def tearDown(self):
        set_language(self.app, self.original_language, persist=False)

    def test_new_install_prefers_russian_and_setting_is_persisted(self):
        with TemporaryDirectory() as scratch:
            settings = QSettings(str(Path(scratch) / "language.ini"), QSettings.Format.IniFormat)
            self.assertEqual(saved_language(settings), "ru")
            set_language(self.app, "en", settings=settings)
            self.assertEqual(saved_language(settings), "en")
            set_language(self.app, "ru", settings=settings)
            self.assertEqual(saved_language(settings), "ru")
            settings.setValue("appearance/language", "other")
            self.assertEqual(saved_language(settings), "ru")

    def test_live_widget_translation_is_reversible_and_preserves_inputs(self):
        root = QWidget()
        self.addCleanup(root.close)
        layout = QVBoxLayout(root)
        label = QLabel("Overview", root)
        button = QPushButton("Training", root)
        button.setToolTip('<p style="max-width: 180px;">Overview</p>')
        group = QGroupBox("Training", root)
        edit = QLineEdit(root)
        edit.setText("Training")
        edit.setPlaceholderText("Overview")
        tabs = QTabWidget(root)
        tabs.addTab(QWidget(), "Overview")
        combo = QComboBox(root)
        combo.addItem("Overview", "general")
        combo.addItem("Training", "training")
        combo.setProperty("_i18n_combo_sources", ["Overview", "Training"])
        combo.setCurrentIndex(1)
        changes = []
        combo.currentTextChanged.connect(changes.append)
        action = QAction("Training", root)
        root.addAction(action)
        for widget in (label, button, group, edit, tabs, combo):
            layout.addWidget(widget)

        set_language(self.app, "ru", persist=False)
        self.assertEqual(label.text(), "Обзор")
        self.assertEqual(button.text(), "Обучение")
        self.assertIn("Обзор", button.toolTip())
        self.assertEqual(group.title(), "Обучение")
        self.assertEqual(action.text(), "Обучение")
        self.assertEqual(tabs.tabText(0), "Обзор")
        self.assertEqual(edit.placeholderText(), "Обзор")
        self.assertEqual(edit.text(), "Training")
        self.assertEqual(combo.itemText(1), "Обучение")
        self.assertEqual(combo.currentData(), "training")
        self.assertEqual(changes, [])

        label.setText("Training")
        retranslate_tree(root)
        self.assertEqual(label.text(), "Обучение")
        set_language(self.app, "en", persist=False)
        self.assertEqual(label.text(), "Training")
        self.assertEqual(combo.itemText(1), "Training")
        self.assertEqual(combo.currentData(), "training")
        self.assertEqual(changes, [])

    def test_new_dialog_is_translated_when_shown(self):
        set_language(self.app, "ru", persist=False)
        dialog = QDialog()
        self.addCleanup(dialog.close)
        dialog.setWindowTitle("Training")
        label = QLabel("Overview", dialog)
        QVBoxLayout(dialog).addWidget(label)
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.windowTitle(), "Обучение")
        self.assertEqual(label.text(), "Обзор")

    def test_navigation_language_control_emits_choice(self):
        navigation = WorkflowNavigation()
        self.addCleanup(navigation.close)
        requested = []
        navigation.language_requested.connect(requested.append)
        navigation.set_language("ru")
        self.assertTrue(navigation.russian_button.isChecked())
        set_language(self.app, "ru", persist=False)
        navigation.set_pages({"general", "training"})
        self.assertEqual(navigation._items["general"].text(), "Обзор")
        self.assertEqual(navigation.russian_button.text(), "Русский")
        self.assertEqual(navigation.english_button.text(), "English")
        navigation.english_button.click()
        self.assertEqual(requested, ["en"])

    def test_dynamic_template_tracks_current_values_and_yields_to_manual_text(self):
        label = QLabel()
        self.addCleanup(label.close)
        set_localized_text(label, "ETA: {eta_str}", eta_str="01:20")
        self.assertEqual(label.text(), "ETA: 01:20")
        set_language(self.app, "ru", persist=False)
        self.assertEqual(label.text(), "Осталось: 01:20")
        set_localized_text(label, "ETA: {eta_str}", eta_str="00:45")
        self.assertEqual(label.text(), "Осталось: 00:45")
        set_language(self.app, "en", persist=False)
        self.assertEqual(label.text(), "ETA: 00:45")

        label.setText("Manual status")
        set_language(self.app, "ru", persist=False)
        self.assertEqual(label.text(), "Manual status")

    def test_dynamic_template_does_not_translate_user_filename(self):
        label = QLabel()
        self.addCleanup(label.close)
        set_language(self.app, "ru", persist=False)
        set_localized_text(label, 'Detecting scenes in "{filename}"', filename="Training")
        self.assertIn('«Training»', label.text())
        set_language(self.app, "en", persist=False)
        self.assertEqual(label.text(), 'Detecting scenes in "Training"')

    def test_formatted_catalog_labels_and_repeated_fields_switch_both_ways(self):
        label = QLabel("Text Encoder 2")
        self.addCleanup(label.close)
        tooltip = "The base LoRA to train on. Leave empty to create a new LoRA"
        label.setToolTip(tooltip)
        set_language(self.app, "ru", persist=False)
        self.assertEqual(label.text(), "Текстовый энкодер 2")
        self.assertEqual(label.toolTip(), "Базовый адаптер LoRA для дообучения. Оставьте поле пустым, чтобы создать новый LoRA.")
        self.assertEqual(
            translate("The base LoRA to train on. Leave empty to create a new LoHa"),
            "The base LoRA to train on. Leave empty to create a new LoHa",
        )
        set_language(self.app, "en", persist=False)
        self.assertEqual(label.text(), "Text Encoder 2")
        self.assertEqual(label.toolTip(), tooltip)


if __name__ == "__main__":
    unittest.main()
