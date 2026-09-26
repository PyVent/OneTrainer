"""Small, runtime-switchable UI translation layer for the Qt applications.

The English text in widget constructors is the source language.  Translations
only affect presentation: config values and editable user text stay untouched.
"""

import re
from functools import lru_cache
from html import escape, unescape
from importlib import import_module
from string import Formatter

from PySide6.QtCore import QEvent, QLibraryInfo, QObject, QSettings, QSignalBlocker, QTimer, QTranslator
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QTabWidget,
    QTextEdit,
    QWidget,
)
from shiboken6 import isValid

LANGUAGES = ("ru", "en")
_SETTING = "appearance/language"
_CORE_RU = {
    "Language": "Язык",
    "LANGUAGE": "ЯЗЫК",
    "Russian": "Русский",
    "English": "Английский",
    "Overview": "Обзор",
    "Training": "Обучение",
    "Prepare": "Подготовка",
    "Train": "Обучение",
    "Generate": "Генерация",
    "Utilities": "Инструменты",
    "APPEARANCE": "ОФОРМЛЕНИЕ",
    "Light": "Светлая",
    "Dark": "Тёмная",
    "Help && Docs": "Справка и документация",
    "Train your vision": "Воплощайте свои идеи",
    "EPOCH": "ЭПОХА",
    "STEP": "ШАГ",
    "SECOND": "СЕКУНДА",
    "MINUTE": "МИНУТА",
    "HOUR": "ЧАС",
    "NEVER": "НИКОГДА",
    "ALWAYS": "ВСЕГДА",
}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    result = dict(_CORE_RU)
    for suffix in ("main", "tools"):
        module_name = f"modules.util.ui.pyside6_i18n_catalog_{suffix}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        result.update(module.RU)
    return result


def language_settings() -> QSettings:
    return QSettings("OneTrainer", "OneTrainer")


def saved_language(settings: QSettings | None = None) -> str:
    value = (settings or language_settings()).value(_SETTING, "ru")
    return value if value in LANGUAGES else "ru"


def _install_qt_base_language(app: QApplication, language: str) -> None:
    previous = getattr(app, "_onetrainer_qt_base_translator", None)
    if previous is not None:
        app.removeTranslator(previous)
        app._onetrainer_qt_base_translator = None
    if language == "ru":
        translator = QTranslator(app)
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if translator.load("qtbase_ru", directory):
            app.installTranslator(translator)
            app._onetrainer_qt_base_translator = translator


def current_language() -> str:
    app = QApplication.instance()
    value = app.property("onetrainerLanguage") if app is not None else None
    # Direct QApplication construction is common in tests and auxiliary tools.
    # Only create_application/install_i18n opts in to the saved RU-first UI.
    return value if value in LANGUAGES else "en"


@lru_cache(maxsize=1)
def _formatted_catalog() -> list[tuple[re.Pattern, str, int]]:
    """Compile catalog keys containing named fields, most specific first."""
    entries = []
    for source, translated in _catalog().items():
        try:
            parts = list(Formatter().parse(source))
        except ValueError:
            continue
        fragments = []
        seen: set[str] = set()
        field_count = 0
        literal_length = 0
        valid = True
        for literal, field, format_spec, conversion in parts:
            fragments.append(re.escape(literal))
            literal_length += len(literal)
            if field is None:
                continue
            if not field.isidentifier() or format_spec or conversion:
                valid = False
                break
            field_count += 1
            if field in seen:
                fragments.append(f"(?P={field})")
            else:
                fragments.append(f"(?P<{field}>.+?)")
                seen.add(field)
        if valid and field_count:
            entries.append((re.compile("^" + "".join(fragments) + "$", re.DOTALL),
                            translated, literal_length))
    entries.sort(key=lambda entry: entry[2], reverse=True)
    return entries


@lru_cache(maxsize=4096)
def _translated_formatted(text: str) -> str | None:
    for pattern, translated, _ in _formatted_catalog():
        match = pattern.fullmatch(text)
        if match:
            try:
                return translated.format(**match.groupdict())
            except (KeyError, ValueError):
                continue
    return None


def _translated_plain(text: str, language: str) -> str:
    if language == "en" or not text:
        return text
    catalog = _catalog()
    if text in catalog:
        return catalog[text]
    stripped = text.strip()
    replacement = catalog.get(stripped)
    if replacement is None:
        replacement = _translated_formatted(stripped)
    return text.replace(stripped, replacement, 1) if replacement is not None else text


def _translate_to(text: str, language: str) -> str:
    if not isinstance(text, str) or language == "en":
        return text
    if text in _catalog():
        return _catalog()[text]
    if not re.search(r"</?[a-z][^>]*>", text, re.IGNORECASE):
        return _translated_plain(text, language)
    # Tooltips built by the form helpers wrap their content in simple HTML.
    # Translate visible text nodes while leaving tags, attributes and markup intact.
    return "".join(
        part if part.startswith("<") else escape(_translated_plain(unescape(part), language))
        for part in re.split(r"(<[^>]+>)", text)
    )


def translate(text: str) -> str:
    """Return presentation text in the application's selected language."""
    return _translate_to(text, current_language())


@lru_cache(maxsize=1)
def _reverse_catalog() -> dict[str, str]:
    result: dict[str, str] = {}
    for source, translated in _catalog().items():
        result.setdefault(translated, source)
    return result


def _source_text(text: str) -> str:
    return _reverse_catalog().get(text, text)


def _render_template(template: str, kwargs: dict, language: str) -> str:
    # Substitutions may contain user file names, paths or captions that happen
    # to match a catalog entry. They must remain byte-for-byte unchanged.
    return _translate_to(template, language).format(**kwargs)


def set_localized_text(widget: QLabel | QAbstractButton, english_template: str, **kwargs) -> None:
    """Set changing text without losing its English source on language switch."""
    if not isinstance(widget, (QLabel, QAbstractButton)):
        raise TypeError("set_localized_text requires a QLabel or QAbstractButton")
    widget.setProperty("_onetrainer_i18n_template", english_template)
    widget.setProperty("_onetrainer_i18n_kwargs", kwargs)
    widget.setProperty("_onetrainer_i18n_text_source", None)
    rendered = _render_template(english_template, kwargs, current_language())
    widget.setProperty("_onetrainer_i18n_rendered", rendered)
    widget.setText(rendered)


def _retranslate_template(widget: QLabel | QAbstractButton, language: str) -> bool:
    template = widget.property("_onetrainer_i18n_template")
    if not isinstance(template, str):
        return False
    if widget.text() != widget.property("_onetrainer_i18n_rendered"):
        # A controller took ownership of this caption after the helper set it.
        for name in ("template", "kwargs", "rendered"):
            widget.setProperty(f"_onetrainer_i18n_{name}", None)
        return False
    kwargs = widget.property("_onetrainer_i18n_kwargs") or {}
    rendered = _render_template(template, kwargs, language)
    widget.setProperty("_onetrainer_i18n_rendered", rendered)
    if widget.text() != rendered:
        widget.setText(rendered)
    return True


def _translate_attribute(obj: QObject, name: str, getter, setter, language: str) -> None:
    current = getter()
    if not current:
        return
    property_name = f"_onetrainer_i18n_{name}_source"
    source = obj.property(property_name)
    if source is None:
        source = _source_text(current)
    elif current not in (source, _translate_to(source, "ru")):
        # A controller changed a dynamic caption since the last pass.
        source = _source_text(current)
    if obj.property(property_name) != source:
        obj.setProperty(property_name, source)
    translated = _translate_to(source, language)
    if current != translated:
        setter(translated)


def _retranslate_object(obj: QObject, language: str) -> None:
    if isinstance(obj, QWidget):
        for name in ("windowTitle", "toolTip", "statusTip", "whatsThis", "accessibleName"):
            _translate_attribute(obj, name, getattr(obj, name), getattr(obj, f"set{name[0].upper()}{name[1:]}"), language)
    elif isinstance(obj, QAction):
        for name in ("text", "toolTip", "statusTip", "whatsThis"):
            _translate_attribute(obj, name, getattr(obj, name), getattr(obj, f"set{name[0].upper()}{name[1:]}"), language)

    if (isinstance(obj, (QLabel, QAbstractButton))
            and not obj.property("_onetrainer_i18n_keep_native")
            and not _retranslate_template(obj, language)):
        _translate_attribute(obj, "text", obj.text, obj.setText, language)
    if isinstance(obj, (QGroupBox, QMenu)):
        _translate_attribute(obj, "title", obj.title, obj.setTitle, language)
    if isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox)):
        _translate_attribute(obj, "placeholderText", obj.placeholderText, obj.setPlaceholderText, language)

    if isinstance(obj, QComboBox):
        sources = obj.property("_i18n_combo_sources")
        if isinstance(sources, list) and len(sources) == obj.count():
            with QSignalBlocker(obj):
                for index, source in enumerate(sources):
                    display = _translate_to(source, language)
                    if obj.itemText(index) != display:
                        obj.setItemText(index, display)

    if isinstance(obj, QTabWidget):
        for index in range(obj.count()):
            page = obj.widget(index)
            if page is None:
                continue
            for name, getter, setter in (
                ("tabText", obj.tabText, obj.setTabText),
                ("tabToolTip", obj.tabToolTip, obj.setTabToolTip),
            ):
                current = getter(index)
                if not current:
                    continue
                property_name = f"_onetrainer_i18n_{name}_source"
                source = page.property(property_name)
                if source is None or current not in (source, _translate_to(source, "ru")):
                    source = _source_text(current)
                    page.setProperty(property_name, source)
                display = _translate_to(source, language)
                if current != display:
                    setter(index, display)


def retranslate_tree(root: QObject) -> None:
    """Translate all visible copy owned by a widget (including its actions)."""
    language = current_language()
    objects = [root, *root.findChildren(QObject)]
    seen: set[int] = set()
    for obj in objects:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        _retranslate_object(obj, language)
        if isinstance(obj, QWidget):
            for action in obj.actions():
                if id(action) not in seen:
                    seen.add(id(action))
                    _retranslate_object(action, language)


class _LanguageShowFilter(QObject):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._pending: dict[int, QWidget] = {}
        self._scheduled = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget):
            window = watched.window()
            self._pending[id(window)] = window
            if not self._scheduled:
                self._scheduled = True
                QTimer.singleShot(0, self._flush)
        return False

    def _flush(self) -> None:
        pending, self._pending = self._pending, {}
        self._scheduled = False
        for window in pending.values():
            if isValid(window):
                retranslate_tree(window)


def install_i18n(app: QApplication) -> None:
    """Restore language and localize dialogs/windows shown after startup."""
    if app.property("_onetrainer_i18n_installed"):
        return
    app.setProperty("onetrainerLanguage", saved_language())
    _install_qt_base_language(app, current_language())
    event_filter = _LanguageShowFilter(app)
    app.installEventFilter(event_filter)
    # Keep the Python wrapper alive for the lifetime of QApplication.
    app._onetrainer_i18n_filter = event_filter
    app.setProperty("_onetrainer_i18n_installed", True)
    for window in app.topLevelWidgets():
        retranslate_tree(window)


def set_language(app: QApplication, language: str, *, persist: bool = True,
                 settings: QSettings | None = None) -> None:
    """Switch every open Qt window and remember the selection if requested."""
    if language not in LANGUAGES:
        raise ValueError(f"Unknown UI language: {language}")
    if not app.property("_onetrainer_i18n_installed"):
        install_i18n(app)
    app.setProperty("onetrainerLanguage", language)
    _install_qt_base_language(app, language)
    if persist:
        (settings or language_settings()).setValue(_SETTING, language)
    for window in app.topLevelWidgets():
        retranslate_tree(window)
        # Non-QObject content such as a Matplotlib canvas can refresh itself.
        QApplication.sendEvent(window, QEvent(QEvent.Type.LanguageChange))
