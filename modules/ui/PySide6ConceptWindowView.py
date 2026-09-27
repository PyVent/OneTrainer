import copy
import math
import threading
from pathlib import Path

from modules.ui.BaseConceptWindowView import BaseConceptWindowView
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util.caption_util import caption_format, caption_key, caption_sample_count
from modules.util.ui import pyside6_components as components
from modules.util.ui.pyside6_i18n import set_localized_text, translate
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.validation_helpers import validate_resolution

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class NonZoomingFigureCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        event.ignore()


class _PreviewSignals(QObject):
    ready = Signal(int, object, object, str)


class PySide6ConceptWindowView(BaseConceptWindowView, QDialog):
    """A transactional editor: dataset → captions → images → sampling → checks."""

    def __init__(self, parent, controller: ConceptWindowController, ui_state, image_ui_state, text_ui_state):
        QDialog.__init__(self, parent)
        BaseConceptWindowView.__init__(self, components)
        self.controller = controller
        self._original = controller.concept
        self._original_states = (ui_state, image_ui_state, text_ui_state)
        controller.concept = copy.deepcopy(controller.concept)
        self.ui_state = PySide6UIState(controller.concept)
        self.image_state = PySide6UIState(controller.concept.image)
        self.text_state = PySide6UIState(controller.concept.text)
        self.image_preview_file_index = 0
        self._preview_augmentations = True
        self._image_layout_mode = None
        self._closed = False
        self.bucket_fig = None
        self._controls = {}
        self._preview_requested = False
        self._preview_busy = False
        self._preview_dirty = True
        self._preview_revision = 0
        self._preview_signals = _PreviewSignals(self)
        self._preview_signals.ready.connect(self._finish_preview)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_image_preview)

        self.setWindowTitle("Concept")
        self.resize(1160, 820)
        self.setMinimumSize(760, 600)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel("Concept settings")
        title.setObjectName("pageTitle")
        heading.addWidget(title)
        heading.addStretch()
        enabled = QCheckBox("Use this concept")
        enabled.setChecked(controller.concept.enabled)
        enabled.toggled.connect(self.ui_state.get_var("enabled").set)
        heading.addWidget(enabled)
        outer.addLayout(heading)
        self.tabs = QTabWidget(self)
        body = QHBoxLayout()
        body.addWidget(self.tabs, 1)
        outer.addLayout(body, 1)

        data = self._page("Dataset")
        captions = self._page("Captions")
        images = self._page("Images")
        sampling = self._page("Sampling")
        stats = self._page("Checks")
        self._build_data(data)
        self._caption_form = self._build_captions(captions)
        self._image_form = self._build_images(images)
        self._build_sampling(sampling)
        self._build_stats(stats)
        self._image_scroll = self.tabs.widget(2)
        self._preview_panel = self._build_preview()
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._preview_scroll.setWidget(self._preview_panel)
        self._preview_scroll.hide()
        body.addWidget(self._preview_scroll)

        footer = QHBoxLayout()
        self._status = self._hint("Changes are applied when you save.")
        footer.addWidget(self._status, 1)
        self._preview_button = QPushButton("Preview")
        self._preview_button.clicked.connect(self._toggle_preview)
        self._preview_button.hide()
        footer.addWidget(self._preview_button)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save concept")
        save.setObjectName("primaryButton")
        save.setDefault(True)
        save.clicked.connect(self._ok)
        footer.addWidget(cancel)
        footer.addWidget(save)
        outer.addLayout(footer)
        self.tabs.currentChanged.connect(self._move_preview)

        for state in (self.ui_state, self.image_state, self.text_state):
            for name in state.obj.types:
                if state.obj.types[name] in (str, bool, int, float):
                    state.get_var(name).subscribe(lambda _: self._schedule_preview(), owner=self)
        self._schedule_preview()
        if "file_size" in controller.concept.concept_stats:
            self._update_concept_stats(controller)
        # Scans are explicit; opening a dialog never starts expensive metadata work.

    def _page(self, title):
        scroll = QScrollArea(self.tabs)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(16)
        layout.setColumnStretch(0, 1)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return page

    def _form(self, page):
        form = QWidget()
        layout = QVBoxLayout(form)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        page.layout().addWidget(form, 0, 0, Qt.AlignmentFlag.AlignTop)
        return form

    def _card(self, form, title, help_text=None):
        card = QGroupBox(title)
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 22, 14, 14)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        form.layout().addWidget(card)
        if help_text:
            grid.addWidget(self._hint(help_text), 0, 0, 1, 2)
        return card

    @staticmethod
    def _hint(text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("pageSubtitle")
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def _label(self, card, row, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setMinimumWidth(190)
        label.setMaximumWidth(230)
        card.layout().addWidget(label, row, 0)

    def _entry(self, card, row, text, state, key, **kwargs):
        self._label(card, row, text)
        widget = components.entry(card, row, 1, state, key, **kwargs)
        self._controls[key] = widget
        return widget

    def _choice(self, card, row, text, state, key, values, command=None):
        self._label(card, row, text)
        widget = components.options_kv(card, row, 1, values, state, key, command=command)
        widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(14)
        self._controls[key] = widget
        return widget

    def _number(self, card, row, text, state, key, minimum=0, maximum=1000000, decimals=None):
        self._label(card, row, text)
        widget = QSpinBox() if decimals is None else QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
            widget.setSingleStep(0.05 if maximum <= 1 else 0.1)
        widget.setRange(minimum, maximum)
        var = state.get_var(key)
        widget.setValue(float(var.get()) if decimals is not None else int(var.get()))
        widget.valueChanged.connect(lambda value: var.set(str(value)))
        card.layout().addWidget(widget, row, 1)
        self._controls[key] = widget
        return widget

    def _check(self, card, row, text, state, key):
        widget = components.switch(card, row, 0, state, key, text=text)
        card.layout().addWidget(widget, row, 0, 1, 2)
        self._controls[key] = widget
        return widget

    def _enabled_by(self, state, key, widgets, predicate=bool):
        def update(value):
            for widget in widgets:
                widget.setEnabled(predicate(value))
        state.get_var(key).subscribe(update, owner=self)
        update(state.get_var(key).get())

    def _build_data(self, page):
        form = self._form(page)
        card = self._card(form, "Dataset location", "One concept groups files that share caption, augmentation and sampling rules.")
        self._entry(card, 1, "Name", self.ui_state, "name")
        self._choice(card, 2, "Purpose", self.ui_state, "type", [
            ("Training", "STANDARD"), ("Validation", "VALIDATION"), ("Prior preservation (LoRA)", "PRIOR_PREDICTION"),
        ])
        self._label(card, 3, "Folder or Hugging Face dataset")
        components.path_entry(card, 3, 1, self.ui_state, "path", mode="dir")
        self._check(card, 4, "Include Subdirectories", self.ui_state, "include_subdirectories")
        download = QPushButton("Download Now")
        download.clicked.connect(self.controller.download_dataset_threaded)
        card.layout().addWidget(download, 5, 1)
        self._enabled_by(self.ui_state, "path", [download], lambda value: bool(value) and not Path(value).is_dir())
        flow = self._card(form, "How samples are built")
        flow.layout().addWidget(self._hint(
            "Files → caption examples → image and text augmentations → cache → epoch sampling.\n"
            "With separate captions, each line gets its own augmented copy of the same image."
        ), 0, 0, 1, 2)

    def _build_captions(self, page):
        form = self._form(page)
        card = self._card(form, "Caption source")
        self._choice(card, 0, "Read captions from", self.text_state, "prompt_source", [
            ("TXT beside each image", "sample"), ("One shared TXT file", "concept"), ("Image filename", "filename"),
        ])
        self._label(card, 1, "Shared TXT file")
        path = components.path_entry(card, 1, 1, self.text_state, "prompt_path", mode="file")
        self._enabled_by(self.text_state, "prompt_source", [path], lambda value: value == "concept")
        mode = self._choice(card, 2, "Multiple non-empty lines", self.text_state, "caption_mode", [
            ("Every line is a separate example", "all"), ("Pick one random line (legacy)", "random"),
        ])
        self._enabled_by(self.text_state, "prompt_source", [mode], lambda value: value != "filename")
        card.layout().addWidget(self._hint(
            "2 lines × 1 image = 2 examples in separate mode. Empty lines are ignored. "
            "A missing or empty TXT keeps one example with an empty caption. "
            "A shared TXT applies its lines to every image."
        ), 3, 0, 1, 2)
        card = self._card(form, "Caption format")
        self._choice(card, 0, "Treat captions as", self.text_state, "caption_format", [
            ("Mixed: detect tags or prose", "auto"), ("Tag lists", "tags"), ("Natural language", "text"),
        ])
        card.layout().addWidget(self._hint(
            "Auto detection is approximate. Review each line in the preview and override its type if needed. "
            "Uncertain lines stay unchanged. Overrides follow the caption text, even when lines are reordered."
        ), 1, 0, 1, 2)
        card = self._card(form, "Tag processing", "These operations only affect captions classified as tags. Natural descriptions stay unchanged.")
        self._entry(card, 1, "Tag Delimiter", self.text_state, "tag_delimiter", required=True)
        self._number(card, 2, "Keep first tags (shuffle / dropout)", self.text_state, "keep_tags_count")
        self._check(card, 3, "Shuffle remaining tags", self.text_state, "enable_tag_shuffling")
        self._check(card, 4, "Randomly remove tags", self.text_state, "tag_dropout_enable")
        drop_mode = self._choice(card, 5, "Removal rule", self.text_state, "tag_dropout_mode", [
            ("Each tag independently", "RANDOM"), ("All eligible tags at once", "FULL"), ("Later tags more often", "RANDOM WEIGHTED"),
        ])
        drop_probability = self._number(card, 6, "Removal probability (0–1)", self.text_state, "tag_dropout_probability", maximum=1, decimals=2)
        self._enabled_by(self.text_state, "tag_dropout_enable", [drop_mode, drop_probability])
        self._choice(card, 7, "Exceptions", self.text_state, "tag_dropout_special_tags_mode", [
            ("No exceptions", "NONE"), ("Always keep listed tags", "WHITELIST"), ("Only remove listed tags", "BLACKLIST"),
        ])
        special = self._entry(card, 8, "Tags or TXT/CSV path", self.text_state, "tag_dropout_special_tags")
        regex = self._check(card, 9, "Interpret exceptions as regular expressions", self.text_state, "tag_dropout_special_tags_regex")
        self._enabled_by(self.text_state, "tag_dropout_special_tags_mode", [special, regex], lambda value: value != "NONE")
        card = self._card(form, "Letter case (tags only)")
        self._check(card, 0, "Force Lowercase", self.text_state, "caps_randomize_lowercase")
        self._check(card, 1, "Randomize Capitalization", self.text_state, "caps_randomize_enable")
        probability = self._number(card, 2, "Probability per tag (0–1)", self.text_state, "caps_randomize_probability", maximum=1, decimals=2)
        modes = QWidget()
        modes_layout = QGridLayout(modes)
        modes_layout.setContentsMargins(0, 0, 0, 0)
        mode_boxes = []
        selected = self.controller.concept.text.caps_randomize_mode.split(",")
        for index, (label, value) in enumerate([("ALL CAPS", "capslock"), ("Title Case", "title"), ("First letter", "first"), ("rAnDoM", "random")]):
            check = QCheckBox(label)
            check.setChecked(value in [item.strip() for item in selected])
            mode_boxes.append((check, value))
            modes_layout.addWidget(check, index // 2, index % 2)
            check.toggled.connect(lambda _: self.text_state.get_var("caps_randomize_mode").set(
                ", ".join(value for box, value in mode_boxes if box.isChecked())))
        card.layout().addWidget(modes, 3, 0, 1, 2)
        self._enabled_by(self.text_state, "caps_randomize_enable", [probability, modes])
        return form

    def _build_images(self, page):
        form = self._form(page)
        card = self._card(form, "Crop and geometry")
        self._check(card, 0, "Random crop position", self.image_state, "enable_crop_jitter")
        self._image_operation(card, 1, "Horizontal flip", "flip")
        self._image_operation(card, 2, "Rotation", "rotate", "random_rotate_max_angle", -180, 180, "°")
        card = self._card(form, "Color", "Color adjustments affect RGB. Alpha is preserved for models that support RGBA.")
        for row, (label, key) in enumerate([("Brightness", "brightness"), ("Contrast", "contrast"), ("Saturation", "saturation"), ("Hue", "hue")], start=1):
            self._image_operation(card, row, label, key, f"random_{key}_max_strength", -1, 1 if key == "hue" else 8)
        card = self._card(form, "Resolution and masks")
        self._check(card, 0, "Resolution Override", self.image_state, "enable_resolution_override")
        resolution = self._entry(card, 1, "Pixels: 512 or 512x768", self.image_state, "resolution_override", extra_validate=validate_resolution())
        self._enabled_by(self.image_state, "enable_resolution_override", [resolution])
        self._check(card, 2, "Circular Mask Generation", self.image_state, "enable_random_circular_mask_shrink")
        self._check(card, 3, "Random Rotate and Crop", self.image_state, "enable_random_mask_rotate_crop")
        card.layout().addWidget(self._hint("Mask operations are used when masked training or the model's mask input is enabled."), 4, 0, 1, 2)
        return form

    def _image_operation(self, card, row, title, key, strength=None, minimum=0, maximum=1, suffix=""):
        self._label(card, row, title)
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = components.NoScrollComboBox()
        options = ["Off", "Random", "Fixed"]
        for label in options:
            combo.addItem(translate(label))
        combo.setProperty("_i18n_combo_sources", options)
        random_var = self.image_state.get_var(f"enable_random_{key}")
        fixed_var = self.image_state.get_var(f"enable_fixed_{key}")
        combo.setCurrentIndex(1 if random_var.get() else 2 if fixed_var.get() else 0)
        layout.addWidget(combo, 1)
        value = None
        if strength:
            value = QDoubleSpinBox()
            current = getattr(self.controller.concept.image, strength)
            value.setDecimals(4)
            value.setRange(min(minimum, current), max(maximum, current))
            value.setSingleStep(1 if key == "rotate" else 0.05)
            value.setSuffix(suffix)
            value.setValue(current)
            value.valueChanged.connect(lambda number: self.image_state.get_var(strength).set(str(number)))
            value.setEnabled(combo.currentIndex() != 0)
            value.setToolTip("Random: maximum deviation. Fixed: signed adjustment.")
            layout.addWidget(value, 1)
        def changed(index):
            random_var.set(index == 1)
            fixed_var.set(index == 2)
            if value is not None:
                value.setEnabled(index != 0)
        combo.currentIndexChanged.connect(changed)
        card.layout().addWidget(container, row, 1)
        self._controls[f"image_mode_{key}"] = combo

    def _build_sampling(self, page):
        form = self._form(page)
        card = self._card(form, "Examples per epoch", "Caption lines create examples. Repeats control how often those examples are used.")
        self._choice(card, 1, "Sampling rule", self.ui_state, "balancing_strategy", [("Multiply by repeats", "REPEATS"), ("Fixed example count", "SAMPLES")])
        self._number(card, 2, "Repeats / example count", self.ui_state, "balancing", decimals=2)
        self._number(card, 3, "Loss Weight", self.ui_state, "loss_weight", decimals=3)
        self._sampling_summary = self._hint("Run a dataset check to count examples.")
        card.layout().addWidget(self._sampling_summary, 4, 0, 1, 2)
        card = self._card(form, "Cached augmentation variants", "Variants are reused when caching is enabled. They do not multiply the epoch size. Each separate caption has its own image variants.")
        self._number(card, 1, "Image variants per example", self.ui_state, "image_variations", minimum=1)
        self._number(card, 2, "Text variants per example", self.ui_state, "text_variations", minimum=1)
        self._entry(card, 3, "Seed", self.ui_state, "seed")
        card.layout().addWidget(self._hint("Without caching, augmentation is regenerated on each repetition. With one cached variant, the same result is reused."), 4, 0, 1, 2)

    def _build_stats(self, page):
        form = self._form(page)
        self.concept_stats_tab = page
        buttons = QWidget()
        layout = QGridLayout(buttons)
        layout.setContentsMargins(0, 0, 0, 0)
        self.refresh_basic_stats_button = QPushButton("Count files and captions")
        self.refresh_advanced_stats_button = QPushButton("Check image and video metadata")
        self.cancel_stats_button = QPushButton("Abort Scan")
        self.cancel_stats_button.setEnabled(False)
        self.processing_time = QLabel("—")
        layout.addWidget(self.refresh_basic_stats_button, 0, 0)
        layout.addWidget(self.refresh_advanced_stats_button, 0, 1)
        layout.addWidget(self.cancel_stats_button, 1, 0)
        layout.addWidget(self.processing_time, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.refresh_basic_stats_button.clicked.connect(lambda: self.controller.get_concept_stats_threaded(self, False, 9999))
        self.refresh_advanced_stats_button.clicked.connect(lambda: self.controller.get_concept_stats_threaded(self, True, 9999))
        self.cancel_stats_button.clicked.connect(lambda: self._cancel_concept_stats(self.controller))
        form.layout().addWidget(buttons)
        self._scan_status = self._hint("No scan yet. Counts include files in the selected folder and enabled subfolders.")
        form.layout().addWidget(self._scan_status)
        groups = [
            ("Files", [("Images", "image_count_preview"), ("Videos", "video_count_preview"), ("Caption files", "caption_count_preview"), ("Masks", "mask_count_preview"), ("Total Size", "file_size_preview"), ("Directories", "dir_count_preview")]),
            ("Pairing checks", [("Images with Captions", "image_count_caption_preview"), ("Video with Captions", "video_count_caption_preview"), ("Unpaired Captions", "caption_count_preview_unpaired"), ("Images with Masks", "image_count_mask_preview"), ("Unpaired Masks", "mask_count_preview_unpaired")]),
            ("Image dimensions", [("Max Pixels", "pixel_max_preview"), ("Avg Pixels", "pixel_avg_preview"), ("Min Pixels", "pixel_min_preview"), ("Smallest Buckets", "small_bucket_preview")]),
            ("Video metadata", [("Max Length", "length_max_preview"), ("Avg Length", "length_avg_preview"), ("Min Length", "length_min_preview"), ("Max FPS", "fps_max_preview"), ("Avg FPS", "fps_avg_preview"), ("Min FPS", "fps_min_preview")]),
            ("Caption lengths", [("Max Caption Length", "caption_max_preview"), ("Avg Caption Length", "caption_avg_preview"), ("Min Caption Length", "caption_min_preview")]),
        ]
        self._advanced_stats_cards = []
        for number, (title, fields) in enumerate(groups):
            card = self._card(form, title)
            for row, (label, attr) in enumerate(fields):
                self._label(card, row, label)
                value = self._hint("—")
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                card.layout().addWidget(value, row, 1)
                setattr(self, attr, value)
            if number:
                card.hide()
                self._advanced_stats_cards.append(card)
        self.bucket_fig, self.bucket_ax = plt.subplots(figsize=(7, 2.5))
        self.canvas = NonZoomingFigureCanvas(self.bucket_fig)
        self.canvas.setMinimumHeight(240)
        form.layout().addWidget(self.canvas)
        self.canvas.hide()
        self._style_chart()

    def _build_preview(self):
        panel = QGroupBox("Preview")
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 22, 12, 12)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setFixedHeight(190)
        layout.addWidget(self._image_label)
        self._filename_label = self._hint("")
        layout.addWidget(self._filename_label)
        actions = QHBoxLayout()
        for text, action in [("Previous", self._prev_image_preview), ("New variant", self._new_variant), ("Next", self._next_image_preview)]:
            button = QPushButton(text)
            button.clicked.connect(action)
            actions.addWidget(button)
        layout.addLayout(actions)
        self._caption_selector = components.NoScrollComboBox()
        self._caption_selector.currentIndexChanged.connect(self._select_caption)
        layout.addWidget(self._caption_selector)
        self._caption_override = components.NoScrollComboBox()
        labels = ["Use concept format", "This caption is tags", "This caption is prose"]
        for label, value in zip(labels, [None, "tags", "text"], strict=True):
            self._caption_override.addItem(translate(label), value)
        self._caption_override.setProperty("_i18n_combo_sources", labels)
        self._caption_override.currentIndexChanged.connect(self._override_caption)
        layout.addWidget(self._caption_override)
        layout.addWidget(self._hint("Original caption"))
        self._source_caption = self._text_preview(90)
        layout.addWidget(self._source_caption)
        self._aug_checkbox = QCheckBox("Show Augmentations")
        self._aug_checkbox.setChecked(True)
        self._aug_checkbox.toggled.connect(self._on_aug_toggle)
        layout.addWidget(self._aug_checkbox)
        self._caption_box = self._text_preview(100)
        layout.addWidget(self._caption_box)
        self._example_summary = self._hint("")
        layout.addWidget(self._example_summary)
        layout.addWidget(self._hint("Augmentation preview. Final crop and resolution follow training settings."))
        return panel

    @staticmethod
    def _text_preview(height):
        box = QTextEdit()
        box.setReadOnly(True)
        box.setAcceptRichText(False)
        box.setFixedHeight(height)
        box.setMinimumWidth(0)
        box.setProperty("_onetrainer_i18n_keep_native", True)
        return box

    def _schedule_preview(self):
        if not self._closed:
            self._preview_revision += 1
            self._preview_dirty = True
            self._update_sampling_summary()
            if not self._preview_scroll.isHidden():
                self._preview_timer.start(300)

    def _move_preview(self, index=None):
        if not hasattr(self, "_preview_button"):
            return
        index = self.tabs.currentIndex()
        wide = self.width() >= 1080
        relevant = index in (1, 2)
        self._image_layout_mode = "side_by_side" if wide else "toggle"
        self._preview_button.setVisible(relevant and not wide)
        self._preview_scroll.setVisible(relevant and (wide or self._preview_requested))
        self.tabs.setVisible(wide or not relevant or not self._preview_requested)
        self._preview_scroll.setMinimumWidth(340 if wide else 0)
        self._preview_scroll.setMaximumWidth(370 if wide else 16777215)
        self._preview_panel.setMaximumWidth(16777215)
        set_localized_text(self._preview_button, "Back to settings" if self._preview_requested else "Preview")
        if self._preview_scroll.isHidden():
            self._preview_timer.stop()
        elif self._preview_dirty:
            self._preview_timer.start(0)

    def _toggle_preview(self):
        self._preview_requested = not self._preview_requested
        self._move_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move_preview()

    def _on_aug_toggle(self, checked):
        self._preview_augmentations = checked
        self._update_image_preview()

    def _prev_image_preview(self):
        self.image_preview_file_index = max(0, self.image_preview_file_index - 1)
        self.controller.preview_caption_index = 0
        self._update_image_preview()

    def _next_image_preview(self):
        self.image_preview_file_index += 1
        self.controller.preview_caption_index = 0
        self._update_image_preview()

    def _new_variant(self):
        self.controller.preview_seed += 1
        self._update_image_preview()

    def _select_caption(self, index):
        if index >= 0:
            self.controller.preview_caption_index = index
            self._update_image_preview()

    def _override_caption(self, index):
        captions = self.controller.preview_captions
        if not captions:
            return
        key = caption_key(captions[self.controller.preview_caption_index])
        overrides = self.controller.concept.text.caption_overrides
        value = self._caption_override.itemData(index)
        if value:
            overrides[key] = value
        else:
            overrides.pop(key, None)
        self._update_image_preview()

    def _update_image_preview(self):
        if self._closed:
            return
        self._preview_timer.stop()
        self._preview_revision += 1
        if self._preview_busy:
            return
        self._preview_busy = True
        self._preview_dirty = False
        revision = self._preview_revision
        # The worker owns a snapshot. Editing settings while it reads a large
        # dataset cannot change an in-flight augmentation or its caption rules.
        worker = copy.copy(self.controller)
        worker.concept = copy.deepcopy(self.controller.concept)
        worker._preview_paths = list(self.controller._preview_paths)
        index, augment = self.image_preview_file_index, self._preview_augmentations
        signals = self._preview_signals

        def run():
            result, error = None, ""
            try:
                result = worker.get_preview_image(index, augment)
            except Exception as exc:
                error = str(exc)
            try:
                signals.ready.emit(revision, worker, result, error)
            except RuntimeError:
                worker.close_preview()  # The owning dialog was destroyed meanwhile.

        threading.Thread(target=run, name="concept-preview", daemon=True).start()

    def _finish_preview(self, revision, worker, result, error):
        self._preview_busy = False
        if self._closed:
            worker.close_preview()
            return
        # Retain the incremental iterator when only augmentation settings changed.
        if worker._preview_paths_key == (self.controller.concept.path, self.controller.concept.include_subdirectories):
            for attr in ("_preview_paths_key", "_preview_paths", "_preview_iterator", "_preview_exhausted"):
                setattr(self.controller, attr, getattr(worker, attr))
        else:
            worker.close_preview()
        if revision != self._preview_revision:
            self._preview_dirty = True
            if not self._preview_scroll.isHidden() and not self._preview_timer.isActive():
                self._preview_timer.start(0)
            return
        if error:
            self.controller.preview_captions = []
            self.controller.preview_error = error
            self._caption_box.setPlainText(error)
            self._update_caption_choices()
            return
        for attr in ("preview_captions", "preview_caption_index", "preview_error", "preview_is_placeholder"):
            setattr(self.controller, attr, getattr(worker, attr))
        image, filename, caption = result
        self._image_label.setPixmap(self._preview_pixmap(image))
        self._set_preview_filename(filename)
        self._caption_box.setPlainText(caption)
        self._update_caption_choices()
        self._update_sampling_summary()

    def _update_caption_choices(self):
        captions = self.controller.preview_captions
        settings = self.controller.concept.text.to_dict()
        selected = self.controller.preview_caption_index
        with QSignalBlocker(self._caption_selector), QSignalBlocker(self._caption_override):
            self._caption_selector.clear()
            for index, text in enumerate(captions):
                kind = translate("Tags") if caption_format(text, settings) == "tags" else translate("Prose")
                self._caption_selector.addItem(translate("Line {number} · {kind}").format(number=index + 1, kind=kind))
            self._caption_selector.setProperty("_onetrainer_i18n_keep_native", True)
            self._caption_selector.setCurrentIndex(selected)
            original = captions[selected] if captions else ""
            self._source_caption.setPlainText(original)
            override = settings["caption_overrides"].get(caption_key(original))
            self._caption_override.setCurrentIndex({"tags": 1, "text": 2}.get(override, 0))
            self._caption_override.setEnabled(bool(captions))
        if self.controller.preview_error:
            self._example_summary.setText(self.controller.preview_error)
        elif captions:
            set_localized_text(self._example_summary, "{lines} caption lines → {examples} examples for this image before repeats.",
                               lines=len(captions), examples=caption_sample_count(captions, settings))
        else:
            set_localized_text(self._example_summary, "No images in this concept yet")

    def _set_preview_filename(self, filename):
        placeholder = filename == "No images in this concept yet"
        self._filename_label.setProperty("_onetrainer_i18n_keep_native", not placeholder)
        if placeholder:
            set_localized_text(self._filename_label, filename)
        else:
            self._filename_label.setProperty("_onetrainer_i18n_template", None)
            self._filename_label.setText(filename)

    def _preview_pixmap(self, image):
        if getattr(self.controller, "preview_is_placeholder", False):
            pixmap = QPixmap(str(Path(__file__).resolve().parents[2] / "resources/icons/icon.png"))
        else:
            pixmap = QPixmap.fromImage(ImageQt(image.convert("RGBA")))
        return pixmap.scaled(300, 190, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def _update_sampling_summary(self):
        concept = self.controller.concept
        stats = concept.concept_stats
        key = "image_caption_examples" if concept.text.caption_mode == "all" else "image_count"
        count = stats.get(key)
        if count is None or not stats.get("scan_complete", True) or stats.get("force_cancelled") or stats.get("caption_source") != [concept.path, concept.include_subdirectories, concept.text.prompt_source, concept.text.prompt_path]:
            set_localized_text(self._sampling_summary, "Run a dataset check to count examples.")
            return
        files = stats.get("image_count", 0)
        if self.controller.train_config.model_type.is_video_model():
            files += stats.get("video_count", 0)
            count += stats.get("video_caption_examples" if concept.text.caption_mode == "all" else "video_count", 0)
        samples = math.floor(count * concept.balancing) if str(concept.balancing_strategy) == "REPEATS" else int(concept.balancing)
        if not concept.enabled or not count:
            samples = 0
        set_localized_text(self._sampling_summary, "{files} supported files → {examples} examples → {samples} samples per epoch before batching.",
                           files=files, examples=count, samples=samples)

    def _update_concept_stats(self, controller):
        if self._closed or self.bucket_fig is None:
            return
        super()._update_concept_stats(controller)
        stats = controller.concept.concept_stats
        advanced = stats.get("image_with_caption_count") != "-"
        for card in self._advanced_stats_cards:
            card.setVisible(advanced)
        self.canvas.setVisible(any(stats.get("aspect_buckets", {}).values()))
        status = "Scan complete. Rescan after changing files or caption sources."
        if stats.get("force_cancelled"):
            status = "Partial scan: counts are incomplete."
        elif not stats.get("scan_complete", True):
            status = "Scanning…"
        set_localized_text(self._scan_status, status)
        if stats.get("scan_complete", True):
            self._style_chart()
        self._update_sampling_summary()

    def _style_chart(self):
        self.text_color = self.palette().text().color().name()
        background = self.palette().window().color().name()
        self.bucket_fig.set_facecolor(background)
        self.bucket_ax.set_facecolor(background)
        self.bucket_ax.tick_params(colors=self.text_color)
        for spine in self.bucket_ax.spines.values():
            spine.set_color(self.text_color)
        self.bucket_fig.tight_layout()

    def _disable_scan_buttons(self):
        super()._disable_scan_buttons()
        self.cancel_stats_button.setEnabled(True)
        set_localized_text(self._scan_status, "Scanning…")

    def _enable_scan_buttons(self):
        if self._closed:
            return
        super()._enable_scan_buttons()
        self.cancel_stats_button.setEnabled(False)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.LanguageChange, QEvent.Type.PaletteChange) and self.bucket_fig is not None:
            if "file_size" in self.controller.concept.concept_stats:
                self._update_concept_stats(self.controller)
            self._style_chart()
            if hasattr(self, "_caption_selector"):
                self._update_caption_choices()

    def _cleanup(self):
        self._closed = True
        self._preview_timer.stop()
        self.controller.cancel_scan_flag.set()
        self.controller.preview_cancel_flag.set()
        if not self._preview_busy:
            self.controller.close_preview()
        if self.bucket_fig is not None:
            plt.close(self.bucket_fig)
            self.bucket_fig = None

    def done(self, result):
        self._cleanup()
        super().done(result)

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)

    def _ok(self):
        for field in [*self.findChildren(QSpinBox), *self.findChildren(QDoubleSpinBox)]:
            field.interpretText()
        for field in self.findChildren(QLineEdit):
            validator = getattr(field, "_validator", None)
            if validator is not None and field.isEnabled():
                error = validator.flush_for_save()
                if error:
                    self._status.setText(error)
                    field.setFocus()
                    return
        text = self.controller.concept.text
        if text.caps_randomize_enable and not text.caps_randomize_mode.strip():
            set_localized_text(self._status, "Choose at least one capitalization style.")
            return
        self._original.from_dict(self.controller.concept.to_dict())
        for state, obj in zip(self._original_states, (self._original, self._original.image, self._original.text), strict=True):
            state.update(obj)
        self.accept()
