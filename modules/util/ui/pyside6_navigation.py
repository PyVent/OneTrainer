"""Workflow navigation, branding and the appearance switch."""

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from modules.util.ui.pyside6_theme import COLORS


def _icon(name: str, color: str) -> QIcon:
    """Draw small monochrome icons in Qt for either colour scheme."""
    pixmap = QPixmap(44, 44)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(2, 2)
    painter.setPen(QPen(QColor(color), 1.7, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p = painter

    if name == "general":
        p.drawPolyline(QPolygonF([QPointF(3, 10), QPointF(11, 3), QPointF(19, 10)]))
        p.drawRect(QRectF(5, 10, 12, 9)); p.drawRect(QRectF(10, 14, 3, 5))
    elif name == "model":
        p.drawPolygon(QPolygonF([QPointF(11, 2), QPointF(19, 6), QPointF(11, 10), QPointF(3, 6)]))
        p.drawPolyline(QPolygonF([QPointF(3, 6), QPointF(3, 15), QPointF(11, 20), QPointF(19, 15), QPointF(19, 6)]))
        p.drawLine(11, 10, 11, 20)
    elif name == "data":
        p.drawRoundedRect(QRectF(5, 2, 13, 18), 2, 2)
        p.drawLine(8, 7, 15, 7); p.drawLine(8, 11, 15, 11); p.drawLine(8, 15, 13, 15)
    elif name == "concepts":
        p.drawPolygon(QPolygonF([QPointF(11, 2), QPointF(19, 7), QPointF(16, 17),
                                 QPointF(11, 20), QPointF(6, 17), QPointF(3, 7)]))
        p.drawEllipse(QRectF(9, 8, 4, 4))
    elif name in ("training", "embedding", "additional embeddings"):
        p.drawRoundedRect(QRectF(4, 4, 15, 15), 2, 2)
        p.drawLine(8, 2, 8, 6); p.drawLine(15, 2, 15, 6); p.drawLine(4, 9, 19, 9)
        p.drawLine(8, 13, 10, 15); p.drawLine(10, 15, 15, 11)
    elif name == "LoRA":
        p.drawEllipse(QRectF(3, 3, 16, 16)); p.drawArc(QRectF(7, 7, 8, 8), 40 * 16, 250 * 16)
        p.drawLine(13, 5, 15, 7)
    elif name == "backup":
        p.drawRoundedRect(QRectF(4, 7, 15, 12), 2, 2)
        p.drawLine(4, 10, 19, 10); p.drawLine(9, 14, 14, 14)
        p.drawLine(7, 7, 7, 4); p.drawLine(7, 4, 16, 4); p.drawLine(16, 4, 16, 7)
    elif name == "sampling":
        p.drawRect(QRectF(6, 3, 11, 15)); p.drawLine(9, 7, 14, 7)
        p.drawLine(11, 10, 11, 15)
    elif name == "cloud":
        p.drawArc(QRectF(3, 9, 8, 8), 70 * 16, 215 * 16)
        p.drawArc(QRectF(8, 4, 11, 11), 10 * 16, 180 * 16)
        p.drawArc(QRectF(14, 10, 6, 7), 275 * 16, 165 * 16)
        p.drawLine(6, 18, 18, 18)
    elif name == "tools":
        p.drawLine(4, 18, 17, 5); p.drawLine(6, 4, 18, 16)
        p.drawEllipse(QRectF(2.5, 16.5, 3, 3)); p.drawEllipse(QRectF(16.5, 2.5, 3, 3))
    elif name == "help":
        p.drawEllipse(QRectF(3, 3, 16, 16))
        p.drawText(QRectF(4, 2, 14, 16), Qt.AlignmentFlag.AlignCenter, "?")
    elif name == "light":
        p.drawEllipse(QRectF(8, 8, 6, 6))
        for x1, y1, x2, y2 in ((11, 2, 11, 5), (11, 17, 11, 20), (2, 11, 5, 11),
                                (17, 11, 20, 11), (4, 4, 6, 6), (16, 16, 18, 18),
                                (4, 18, 6, 16), (16, 6, 18, 4)):
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    elif name == "dark":
        p.drawArc(QRectF(4, 3, 15, 16), 115 * 16, 250 * 16)
        p.drawArc(QRectF(9, 1, 12, 15), 95 * 16, 210 * 16)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


class WorkflowNavigation(QFrame):
    page_selected = Signal(str)
    theme_requested = Signal(str)
    help_requested = Signal()

    SECTIONS = (
        ("Prepare", (("general", "Overview"), ("model", "Model"), ("data", "Data"), ("concepts", "Concepts"))),
        ("Train", (("training", "Training"), ("LoRA", "LoRA"), ("embedding", "Embedding"),
                   ("additional embeddings", "Additional embeddings"), ("backup", "Backups"))),
        ("Generate", (("sampling", "Sampling"),)),
        ("Utilities", (("tools", "Tools"), ("cloud", "Cloud"))),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowNavigation")
        self.setMinimumWidth(212)
        self.setMaximumWidth(252)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._theme = "light"
        self._selected: str | None = None
        self._items: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 18, 14, 15)
        outer.setSpacing(12)

        brand = QWidget(self)
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(4, 0, 0, 10)
        brand_layout.setSpacing(10)
        logo = QLabel(brand)
        logo.setPixmap(QPixmap(str(Path(__file__).resolve().parents[3] / "resources/icons/icon.png"))
                       .scaled(43, 43, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation))
        brand_layout.addWidget(logo)
        brand_copy = QWidget(brand)
        brand_text = QVBoxLayout(brand_copy)
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        title = QLabel("OneTrainer", brand_copy)
        title.setObjectName("navBrand")
        brand_text.addWidget(title)
        subtitle = QLabel("Train your vision", brand_copy)
        subtitle.setObjectName("navTagline")
        brand_text.addWidget(subtitle)
        brand_layout.addWidget(brand_copy, 1)
        outer.addWidget(brand)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("navScroll")
        self._list = QWidget(scroll)
        self._list.setObjectName("navList")
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 2, 0)
        self._list_layout.setSpacing(2)
        scroll.setWidget(self._list)
        outer.addWidget(scroll, 1)

        divider = QFrame(self)
        divider.setObjectName("navDivider")
        divider.setFixedHeight(1)
        outer.addWidget(divider)

        appearance = QLabel("APPEARANCE", self)
        appearance.setObjectName("navSection")
        outer.addWidget(appearance)
        theme_row = QWidget(self)
        theme_layout = QHBoxLayout(theme_row)
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(6)
        self.light_button = QPushButton("Light", theme_row)
        self.dark_button = QPushButton("Dark", theme_row)
        for button, mode in ((self.light_button, "light"), (self.dark_button, "dark")):
            button.setObjectName("navTheme")
            button.setCheckable(True)
            button.setAccessibleName(f"Switch to {mode} theme")
            button.clicked.connect(lambda _checked, selected=mode: self.theme_requested.emit(selected))
            theme_layout.addWidget(button, 1)
        outer.addWidget(theme_row)

        self.help_button = QPushButton("Help && Docs", self)
        self.help_button.setObjectName("navHelp")
        self.help_button.clicked.connect(self.help_requested)
        outer.addWidget(self.help_button)
        self.set_theme("light")

    def sizeHint(self) -> QSize:
        return QSize(242, 700)

    def set_pages(self, available: set[str]):
        """Rebuild entries when a model or training method changes."""
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self._items.clear()
        for heading, pages in self.SECTIONS:
            visible = [(key, label) for key, label in pages if key in available]
            if not visible:
                continue
            section = QLabel(heading.upper(), self._list)
            section.setObjectName("navSection")
            self._list_layout.addWidget(section)
            for key, label in visible:
                button = QPushButton(label, self._list)
                button.setObjectName("navPage")
                button.setCheckable(True)
                button.setAccessibleName(label)
                button.clicked.connect(lambda _checked, page=key: self._choose_page(page))
                self._list_layout.addWidget(button)
                self._items[key] = button
            self._list_layout.addSpacing(9)
        self._list_layout.addStretch(1)
        self.select_page(self._selected if self._selected in self._items else None)

    def select_page(self, key: str | None):
        self._selected = key
        for page, button in self._items.items():
            button.setChecked(page == key)
        self._update_icons()

    def _choose_page(self, key: str):
        self.select_page(key)
        self.page_selected.emit(key)

    def set_theme(self, theme: str):
        if theme not in COLORS:
            raise ValueError(theme)
        self._theme = theme
        c = COLORS[theme]
        active_background = (
            "qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #2878f1,stop:1 #4e4de9)"
            if theme == "dark" else
            "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #dce9ff,stop:1 #e4eaff)"
        )
        active_text = "#ffffff" if theme == "dark" else c["accent"]
        self.setStyleSheet(f"""
            QFrame#workflowNavigation {{ background: {c['surface']};
                border-right: 1px solid {c['border']}; }}
            QWidget#navList, QScrollArea#navScroll {{ background: {c['surface']}; border: none; }}
            QLabel#navBrand {{ font-size: 18px; font-weight: 700; color: {c['text']}; }}
            QLabel#navTagline {{ font-size: 11px; color: {c['muted']}; }}
            QLabel#navSection {{ color: {c['muted']}; font-size: 10px;
                font-weight: 700; padding: 6px 6px 5px 6px; }}
            QPushButton#navPage {{ text-align: left; border: none; border-radius: 8px;
                background: transparent; color: {c['text']}; font-size: 12px;
                padding: 7px 9px; min-height: 28px; }}
            QPushButton#navPage:hover {{ background: {c['hover']}; }}
            QPushButton#navPage:checked {{ background: {active_background};
                color: {active_text}; font-weight: 700; }}
            QFrame#navDivider {{ background: {c['border']}; border: none; }}
            QPushButton#navTheme {{ border: 1px solid {c['border']}; border-radius: 7px;
                background: {c['window']}; color: {c['muted']}; padding: 4px; }}
            QPushButton#navTheme:checked {{ background: {c['selection']};
                color: {c['accent']}; border-color: {c['accent']}; font-weight: 700; }}
            QPushButton#navHelp {{ text-align: left; border: none; border-radius: 7px;
                background: transparent; color: {c['muted']}; padding: 6px 8px; }}
            QPushButton#navHelp:hover {{ background: {c['hover']}; color: {c['text']}; }}
        """)
        self.light_button.setChecked(theme == "light")
        self.dark_button.setChecked(theme == "dark")
        self._update_icons()

    def _update_icons(self):
        c = COLORS[self._theme]
        for key, button in self._items.items():
            active_color = "#ffffff" if self._theme == "dark" else c["accent"]
            button.setIcon(_icon(key, active_color if key == self._selected else c["muted"]))
            button.setIconSize(QSize(19, 19))
        self.light_button.setIcon(_icon("light", c["accent"] if self._theme == "light" else c["muted"]))
        self.dark_button.setIcon(_icon("dark", c["accent"] if self._theme == "dark" else c["muted"]))
        self.help_button.setIcon(_icon("help", c["muted"]))
