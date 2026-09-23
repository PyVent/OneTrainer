"""Application colours and the locally stored appearance preference."""

from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


COLORS = {
    "light": {
        "window": "#f7f9fd", "surface": "#ffffff", "input": "#ffffff",
        "text": "#17243d", "muted": "#66758e", "border": "#dbe3ef",
        "hover": "#edf3ff", "accent": "#245eea", "accent_hover": "#174ccc",
        "selection": "#dfeaff", "disabled": "#edf0f5",
        "primary": "#245eea", "primary_hover": "#174ccc",
        "danger_text": "#a94f60", "danger_border": "#e7cbd1",
        "danger_surface": "#fff9fa", "danger_hover": "#fceef1",
        "copy_text": "#314b70", "scroll_handle": "#b9c6d9",
    },
    "dark": {
        "window": "#101827", "surface": "#192437", "input": "#202d42",
        "text": "#e8eef9", "muted": "#a4b2ca", "border": "#33435d",
        "hover": "#263958", "accent": "#6c9aff", "accent_hover": "#87acff",
        "selection": "#27446f", "disabled": "#243147",
        "primary": "#316bdf", "primary_hover": "#447dec",
        "danger_text": "#e8a1ae", "danger_border": "#66505e",
        "danger_surface": "#2a2733", "danger_hover": "#3b2d39",
        "copy_text": "#d6e3f8", "scroll_handle": "#52647f",
    },
}


def theme_settings() -> QSettings:
    return QSettings("OneTrainer", "OneTrainer")


def saved_theme(settings: QSettings | None = None) -> str:
    value = (settings or theme_settings()).value("appearance/theme", "light")
    return value if value in COLORS else "light"


def apply_theme(app: QApplication, theme: str, *, persist: bool = False,
                settings: QSettings | None = None) -> None:
    if theme not in COLORS:
        raise ValueError(f"Unknown theme: {theme}")
    color = COLORS[theme]
    icon_root = (Path(__file__).resolve().parents[3] / "resources" / "icons").as_posix()
    down_arrow = f"{icon_root}/chevron-down-{theme}.svg"
    up_arrow = f"{icon_root}/chevron-up-{theme}.svg"
    check_icon = f"{icon_root}/check-{theme}.svg"
    app.styleHints().setColorScheme(
        Qt.ColorScheme.Dark if theme == "dark" else Qt.ColorScheme.Light
    )

    palette = QPalette()
    for role, value in (
        (QPalette.ColorRole.Window, color["window"]),
        (QPalette.ColorRole.WindowText, color["text"]),
        (QPalette.ColorRole.Base, color["input"]),
        (QPalette.ColorRole.AlternateBase, color["surface"]),
        (QPalette.ColorRole.Text, color["text"]),
        (QPalette.ColorRole.Button, color["surface"]),
        (QPalette.ColorRole.ButtonText, color["text"]),
        (QPalette.ColorRole.Highlight, color["primary"]),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.ToolTipBase, color["surface"]),
        (QPalette.ColorRole.ToolTipText, color["text"]),
        (QPalette.ColorRole.PlaceholderText, color["muted"]),
        (QPalette.ColorRole.Mid, color["border"]),
        (QPalette.ColorRole.Dark, color["border"]),
        (QPalette.ColorRole.Light, color["surface"]),
    ):
        palette.setColor(role, QColor(value))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(color["disabled"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(color["muted"]))
    app.setPalette(palette)

    app.setStyleSheet(f"""
        QWidget {{ color: {color['text']}; }}
        QMainWindow, QDialog {{ background: {color['window']}; }}
        QWidget#trainingTopBar, QWidget#trainingBottomBar {{
            background: {color['surface']};
            border-bottom: 1px solid {color['border']};
        }}
        QLabel#pageHeading {{ font-size: 22px; font-weight: 700; }}
        QLabel#pageSubtitle {{ color: {color['muted']}; font-size: 12px; }}
        QLabel#errorStatus {{ color: {color['danger_text']}; }}
        QLabel#conceptPreview {{
            background: {color['surface']}; border: 1px solid {color['border']};
            border-radius: 8px;
        }}
        QLabel#emptyState {{
            color: {color['muted']}; background: {color['surface']};
            border: 1px dashed {color['border']}; border-radius: 9px;
            padding: 24px; font-size: 13px;
        }}
        QWidget#conceptCard, QWidget#sampleCard {{
            background: {color['surface']}; border: 1px solid {color['border']};
            border-radius: 10px;
        }}
        QGroupBox {{
            background: {color['surface']}; border: 1px solid {color['border']};
            border-radius: 10px; margin-top: 10px; padding-top: 10px;
            font-weight: 600;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 4px; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
            background: {color['input']}; border: 1px solid {color['border']};
            border-radius: 7px; padding: 4px 9px; min-height: 28px;
            selection-background-color: {color['accent']};
        }}
        QComboBox {{ padding-right: 36px; }}
        QComboBox::drop-down {{
            subcontrol-origin: padding; subcontrol-position: top right;
            width: 30px; border-left: 1px solid {color['border']};
        }}
        QComboBox::down-arrow {{ image: url("{down_arrow}"); width: 12px; height: 8px; }}
        QComboBox QAbstractItemView {{
            background: {color['surface']}; color: {color['text']};
            border: 1px solid {color['border']};
            selection-background-color: {color['selection']};
            selection-color: {color['text']};
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 3px 9px; }}
        QSpinBox, QDoubleSpinBox {{ padding-right: 24px; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: top right;
            width: 23px; border-left: 1px solid {color['border']};
            border-bottom: 1px solid {color['border']};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 23px; border-left: 1px solid {color['border']};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url("{up_arrow}"); width: 9px; height: 6px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url("{down_arrow}"); width: 9px; height: 6px;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {color['accent']};
        }}
        QLineEdit[invalid="true"] {{ border-color: {color['danger_text']}; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
            background: {color['disabled']}; color: {color['muted']};
        }}
        QCheckBox::indicator {{
            width: 15px; height: 15px; border: 1px solid {color['border']};
            border-radius: 3px; background: {color['input']};
        }}
        QCheckBox::indicator:checked {{
            background: {color['accent']}; border-color: {color['accent']};
            image: url("{check_icon}");
        }}
        QPushButton, QToolButton {{
            background: {color['surface']}; border: 1px solid {color['border']};
            border-radius: 7px; padding: 4px 11px; min-height: 28px;
        }}
        QPushButton:hover, QToolButton:hover {{
            background: {color['hover']}; border-color: {color['accent']};
        }}
        QPushButton:pressed, QToolButton:pressed {{ background: {color['selection']}; }}
        QPushButton:focus, QToolButton:focus {{ border-color: {color['accent']}; }}
        QPushButton#removeAction, QPushButton#stopTrainingAction {{
            background: {color['danger_surface']}; color: {color['danger_text']};
            border-color: {color['danger_border']}; font-weight: 600;
        }}
        QPushButton#removeAction:hover, QPushButton#stopTrainingAction:hover {{
            background: {color['danger_hover']}; border-color: {color['danger_text']};
        }}
        QPushButton#stopTrainingAction:disabled {{
            background: {color['disabled']}; color: {color['muted']};
            border-color: {color['border']};
        }}
        QPushButton#copyAction {{
            background: {color['surface']}; color: {color['copy_text']};
            border-color: {color['border']};
        }}
        QPushButton#copyAction:hover {{
            background: {color['hover']}; border-color: {color['accent']};
        }}
        QToolButton#presetMenuButton {{ padding-right: 27px; }}
        QToolButton#presetMenuButton::menu-indicator {{
            image: url("{down_arrow}");
            subcontrol-origin: padding; subcontrol-position: center right;
            width: 12px; height: 8px; right: 8px;
        }}
        QMenu {{
            background: {color['surface']}; color: {color['text']};
            border: 1px solid {color['border']}; padding: 4px;
        }}
        QMenu::item {{ padding: 6px 20px; min-height: 22px; border-radius: 5px; }}
        QMenu::item:selected {{
            background: {color['selection']}; color: {color['text']};
        }}
        QMenu::separator {{
            height: 1px; background: {color['border']}; margin: 5px 8px;
        }}
        QPushButton#primaryHeaderAction, QPushButton#startTrainingAction,
        QPushButton#primaryAction {{
            background: {color['primary']}; color: #ffffff;
            border-color: {color['primary']}; font-weight: 600;
        }}
        QPushButton#primaryHeaderAction:hover, QPushButton#startTrainingAction:hover,
        QPushButton#primaryAction:hover {{
            background: {color['primary_hover']}; border-color: {color['primary_hover']};
        }}
        QPushButton:disabled, QToolButton:disabled {{
            background: {color['disabled']}; color: {color['muted']};
            border-color: {color['border']};
        }}
        QTabBar::tab {{
            background: {color['surface']}; color: {color['muted']};
            border: 1px solid {color['border']};
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            padding: 7px 13px; margin-right: 3px;
        }}
        QTabBar::tab:selected {{
            background: {color['input']}; color: {color['text']};
            border-bottom: 2px solid {color['primary']}; font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{ background: {color['hover']}; }}
        QScrollBar:vertical {{
            background: transparent; width: 10px; border: none; margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {color['scroll_handle']}; border-radius: 4px;
            margin: 2px; min-height: 26px;
        }}
        QScrollBar:horizontal {{
            background: transparent; height: 10px; border: none; margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {color['scroll_handle']}; border-radius: 4px;
            margin: 2px; min-width: 26px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            background: transparent; border: none; width: 0; height: 0;
        }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QScrollArea, QTabWidget::pane {{ border: none; }}
        QProgressBar {{
            background: {color['disabled']}; border: 1px solid {color['border']};
            border-radius: 5px; text-align: center;
        }}
        QProgressBar::chunk {{ background: {color['accent']}; border-radius: 4px; }}
        QToolTip {{ background: {color['surface']}; color: {color['text']};
            border: 1px solid {color['border']}; }}
    """)
    app.setProperty("onetrainerTheme", theme)
    if persist:
        (settings or theme_settings()).setValue("appearance/theme", theme)
