from collections.abc import Callable
from pathlib import Path

from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseTrainUIView import BaseTrainUIView
from modules.ui.CloudTabController import CloudTabController
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.LoraTabController import LoraTabController
from modules.ui.ModelTabController import ModelTabController
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.ui.PySide6AdditionalEmbeddingsTabView import PySide6AdditionalEmbeddingsTabView
from modules.ui.PySide6CaptionUIView import PySide6CaptionUIView
from modules.ui.PySide6CloudTabView import PySide6CloudTabView
from modules.ui.PySide6ConceptTabView import PySide6ConceptTabView
from modules.ui.PySide6ConvertModelUIView import PySide6ConvertModelUIView
from modules.ui.PySide6LoraTabView import PySide6LoraTabView
from modules.ui.PySide6ModelTabView import PySide6ModelTabView
from modules.ui.PySide6ProfilingWindowView import PySide6ProfilingWindowView
from modules.ui.PySide6SampleWindowView import PySide6SampleWindowView
from modules.ui.PySide6SamplingTabView import PySide6SamplingTabView
from modules.ui.PySide6TopBarView import PySide6TopBarView
from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
from modules.ui.PySide6VideoToolUIView import PySide6VideoToolUIView
from modules.ui.SamplingTabController import SamplingTabController
from modules.ui.TopBarController import TopBarController
from modules.ui.TrainingTabController import TrainingTabController
from modules.ui.TrainUIController import TrainUIController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_i18n import (
    current_language,
    retranslate_tree,
    set_language,
    set_localized_text,
)
from modules.util.ui.pyside6_i18n import (
    translate as tr,
)
from modules.util.ui.pyside6_navigation import WorkflowNavigation
from modules.util.ui.pyside6_theme import apply_theme, saved_theme
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.validation import flush_and_validate_for_save

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PySide6TrainView(BaseTrainUIView, QMainWindow, metaclass=QtABCMeta):
    PAGE_COPY = {
        "general": ("Training Configuration", "Set paths, monitoring and performance options before training."),
        "model": ("Model", "Choose the source model, components and output options."),
        "data": ("Data", "Prepare images and choose how training data is cached."),
        "concepts": ("Concepts", "Organize datasets and their training settings."),
        "training": ("Training", "Tune optimization, scheduling and training behavior."),
        "LoRA": ("LoRA", "Configure adapter training for the selected model."),
        "embedding": ("Embedding", "Configure embedding training and token behavior."),
        "additional embeddings": ("Additional Embeddings", "Manage extra embeddings used during training."),
        "backup": ("Backups", "Control automatic backups and model saves."),
        "sampling": ("Sampling", "Choose when and how preview images are generated."),
        "tools": ("Tools", "Open dataset, video, conversion, sampling and profiling tools."),
        "cloud": ("Cloud", "Configure a remote training instance and connection."),
    }

    def __init__(self):
        QMainWindow.__init__(self)

        train_config = TrainConfig.default_values()
        ui_state = PySide6UIState(train_config)
        controller = TrainUIController(train_config)

        BaseTrainUIView.__init__(self, pyside6_components, controller, ui_state)
        self.controller.view = self

        self.setWindowTitle("OneTrainer")
        self.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[2] / "resources/icons/icon.png")))
        self.resize(1100, 740)

        self.status_label = None
        self.eta_label = None
        self.training_button = None
        self.export_button = None
        self.tabview: QTabWidget | None = None
        self.navigation: WorkflowNavigation | None = None
        self._tab_widgets: dict[str, QWidget] = {}

        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.cloud_tab = None
        self.concepts_tab = None
        self.sampling_tab = None
        self.additional_embeddings_tab = None

        central = QWidget(self)
        self.setCentralWidget(central)
        central_lo = QHBoxLayout(central)
        central_lo.setContentsMargins(0, 0, 0, 0)
        central_lo.setSpacing(0)

        self.navigation = WorkflowNavigation(central)
        central_lo.addWidget(self.navigation)

        main_area = QWidget(central)
        main_layout = QGridLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 1)
        central_lo.addWidget(main_area, 1)

        self.top_bar_component = self._build_top_bar(main_area)
        self.top_bar_component.setObjectName("trainingTopBar")
        main_layout.addWidget(self.top_bar_component, 0, 0)

        page_content = QWidget(main_area)
        page_content_layout = QVBoxLayout(page_content)
        page_content_layout.setContentsMargins(0, 0, 0, 0)
        page_content_layout.setSpacing(0)
        heading = QWidget(page_content)
        heading_layout = QVBoxLayout(heading)
        heading_layout.setContentsMargins(18, 13, 18, 11)
        heading_layout.setSpacing(2)
        self.page_heading = QLabel(heading)
        self.page_heading.setObjectName("pageHeading")
        self.page_subtitle = QLabel(heading)
        self.page_subtitle.setObjectName("pageSubtitle")
        self.page_subtitle.setWordWrap(True)
        heading_layout.addWidget(self.page_heading)
        heading_layout.addWidget(self.page_subtitle)
        page_content_layout.addWidget(heading)

        self.tabview = QTabWidget(page_content)
        self.tabview.tabBar().hide()
        page_content_layout.addWidget(self.tabview, 1)
        main_layout.addWidget(page_content, 1, 0)

        self.navigation.page_selected.connect(self._show_navigation_page)
        self.navigation.theme_requested.connect(self._change_theme)
        self.navigation.language_requested.connect(self._change_language)
        self.navigation.help_requested.connect(self.top_bar_component.open_wiki)
        app = QApplication.instance()
        self.navigation.set_theme(app.property("onetrainerTheme") or saved_theme())
        self.navigation.set_language(current_language())
        self.tabview.currentChanged.connect(self._sync_navigation_selection)

        bottom = self._build_bottom_bar(main_area)
        main_layout.addWidget(bottom, 2, 0)

        self._create_tabs()
        self._refresh_navigation()
        self.change_training_method(self.controller.train_config.training_method)
        self._update_additional_embeddings_tab(self.controller.train_config.model_type)

        self._profiling_controller = ProfilingWindowController()
        self.profiling_window = self._profiling_controller.create_window(self, PySide6ProfilingWindowView)

        self.controller._check_start_always_on_tensorboard()
        self._workspace_dir_var = self.ui_state.get_var("workspace_dir")
        self._workspace_dir_subscription_id = self._workspace_dir_var.subscribe(
            lambda _: self.controller._on_workspace_dir_change_trace(), owner=self
        )

    def closeEvent(self, event):
        if self.controller.training_thread is not None and self.controller.training_thread.is_alive():
            QMessageBox.warning(
                self,
                tr("Training in progress"),
                tr("A training is currently running. Stop the training before closing the window."),
            )
            event.ignore()
            return
        errors = flush_and_validate_for_save(self.ui_state)
        if errors:
            QMessageBox.warning(self, tr("Cannot save configuration"), "\n".join(errors))
            event.ignore()
            return
        self.save_default()
        self.controller._stop_always_on_tensorboard()
        self._workspace_dir_var.unsubscribe(self._workspace_dir_subscription_id)
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_action_labels)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.LanguageChange:
            self._fit_action_labels()

    def _fit_action_labels(self):
        if not getattr(self, "top_bar_component", None):
            return
        self.top_bar_component.fit_action_labels()
        if getattr(self, "training_tab", None):
            self.training_tab.fit_advanced_button_labels()
        bottom = self.findChild(QWidget, "trainingBottomBar")
        if bottom is None:
            return
        layout = bottom.layout()
        for column in (3, 4, 5, 6):
            item = layout.itemAtPosition(0, column)
            button = item.widget() if item is not None else None
            if button is not None:
                button.setMinimumWidth(button.sizeHint().width() + 4)
                button.updateGeometry()
        layout.invalidate()

    # --- BaseTrainUIView abstract method implementations ---

    def on_update_status(self, status: str):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(lambda: self.status_label.setText(tr(status)))

    def on_training_started(self):
        self._set_training_button_style("running")

    def on_training_stopped(self, error_caught: bool):
        self.eta_label.setText("")
        self._set_training_button_style("idle")

    def on_training_stopping(self):
        self._set_training_button_style("stopping")

    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(lambda: self._do_update_progress(epoch_step, max_step, epoch, max_epoch, eta_str))

    def _do_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None):
        self.set_step_progress(epoch_step, max_step)
        self.set_epoch_progress(epoch, max_epoch)
        if eta_str is None:
            self.eta_label.setText("")
        else:
            set_localized_text(self.eta_label, "ETA: {eta_str}", eta_str=eta_str)

    def schedule_on_main_thread(self, fn: Callable):
        # The 3-argument form (msec, context, fn) is thread-safe: Qt marshals the call
        # to the thread where `self` lives (the main thread), unlike the 2-arg form.
        QTimer.singleShot(0, self, fn)

    def get_cloud_reattach(self) -> bool:
        return self.cloud_tab.reattach

    def save_default(self):
        self.top_bar_component.save_default()
        self.concepts_tab.save_current_config()
        self.sampling_tab.save_current_config()
        self.additional_embeddings_tab.save_current_config()

    def show_validation_errors(self, errors: list[str]):
        bullet_list = "\n".join(f"• {e}" for e in errors)
        QMessageBox.critical(
            self, tr("Cannot Start Training"),
            tr("Please fix the following errors before training:\n\n{bullet_list}").format(
                bullet_list=bullet_list,
            ),
        )

    def confirm(self, title: str, message: str) -> bool:
        buttons = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        return QMessageBox.question(self, tr(title), tr(message), buttons) == QMessageBox.StandardButton.Ok

    def open_dataset_tool(self):
        self.wait_window(self.controller.open_dataset_tool(self, PySide6CaptionUIView))

    def open_video_tool(self):
        self.wait_window(self.controller.open_video_tool(self, PySide6VideoToolUIView))

    def open_convert_model_tool(self):
        self.wait_window(self.controller.open_convert_model_tool(self, PySide6ConvertModelUIView))

    def open_sampling_tool(self):
        self.controller.open_sampling_tool(self, PySide6SampleWindowView)

    def open_manual_sample_window(self):
        self.controller.open_manual_sample_window(self, PySide6SampleWindowView)

    def wait_window(self, window):
        window.exec()

    def show_window(self, window):
        window.show()

    def connect_window_closed(self, window, callback):
        window.finished.connect(lambda _: callback())

    # --- PySide6 layout builders ---

    def _build_top_bar(self, master):
        return PySide6TopBarView(
            master,
            TopBarController(self.controller.train_config),
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def _build_bottom_bar(self, parent):
        frame = QWidget(parent)
        frame.setObjectName("trainingBottomBar")
        lo = QGridLayout(frame)
        lo.setColumnStretch(0, 1)
        lo.setColumnStretch(2, 2)

        status_frame = QWidget(frame)
        status_lo = QGridLayout(status_frame)
        status_lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(status_frame, 0, 1)

        self.build_bottom_bar_content(frame, status_frame, self.controller, self.ui_state)
        self._set_training_button_style("idle")
        return frame

    def _create_scrollable_tab(self, configure_fn):
        tab_page = QWidget()
        tab_lo = pyside6_components._layout(tab_page)
        tab_lo.setRowStretch(0, 1)
        tab_lo.setColumnStretch(0, 1)
        scroll, frame = pyside6_components.scrollable_frame(tab_page)
        tab_lo.addWidget(scroll, 0, 0)
        configure_fn(frame)
        return tab_page

    def _configure_general_frame(self, frame):
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout

        lo = pyside6_components._layout(frame)
        sections = (
            (
                "Paths and run safety",
                "Set the workspace and cache locations, then choose how to resume or protect a run.",
                7,
            ),
            (
                "Monitoring and validation",
                "Control TensorBoard access and when validation runs.",
                6,
            ),
            (
                "Devices and performance",
                "Select compute devices and tune data loading, offloading, and gradient reduction.",
                10,
            ),
        )
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setVerticalSpacing(14)
        lo.setColumnStretch(0, 1)
        cards = []
        containers = []
        for card_row, (title, description, count) in enumerate(sections):
            card = QGroupBox(title, frame)
            card.setObjectName("overviewCard")
            body = QVBoxLayout(card)
            body.setContentsMargins(16, 18, 16, 16)
            body.setSpacing(10)

            subtitle = QLabel(description, card)
            subtitle.setObjectName("pageSubtitle")
            subtitle.setWordWrap(True)
            body.addWidget(subtitle)

            field_container = QWidget(card)
            field_grid = QGridLayout(field_container)
            field_grid.setObjectName("overviewFieldGrid")
            field_grid.setContentsMargins(0, 0, 0, 0)
            field_grid.setHorizontalSpacing(22)
            field_grid.setVerticalSpacing(10)
            body.addWidget(field_container)
            containers.append(field_container)
            cards.append((field_grid, count))
            lo.addWidget(card, card_row, 0)
        lo.setRowStretch(len(cards), 1)

        self.build_general_tab_content(*containers, self.controller, self.ui_state)
        cards = [
            (grid, [(grid.itemAtPosition(row, 0).widget(), grid.itemAtPosition(row, 1).widget())
                    for row in range(count)])
            for grid, count in cards
        ]

        scroll = frame.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        active_columns = None

        def arrange_fields():
            nonlocal active_columns
            available_width = scroll.viewport().width() if scroll is not None else frame.width()
            columns = 2 if available_width >= 1180 else 1
            if columns == active_columns:
                return
            for grid, pairs in cards:
                for label, control in pairs:
                    grid.removeWidget(label)
                    grid.removeWidget(control)
                for col in range(4):
                    grid.setColumnStretch(col, 1 if col % 2 and col < columns * 2 else 0)
                for index, (label, control) in enumerate(pairs):
                    row, pair_col = divmod(index, columns)
                    grid.addWidget(label, row, pair_col * 2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    grid.addWidget(control, row, pair_col * 2 + 1)
            active_columns = columns
            lo.invalidate()
            frame.updateGeometry()
            if scroll is not None:
                scroll.widget().layout().invalidate()
                scroll.widget().resize(scroll.viewport().width(), scroll.widget().height())

        class _OverviewResizeFilter(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Resize:
                    arrange_fields()
                return False

        frame._overview_resize_filter = _OverviewResizeFilter(frame)
        frame.installEventFilter(frame._overview_resize_filter)
        if scroll is not None:
            scroll.viewport().installEventFilter(frame._overview_resize_filter)
        arrange_fields()

    def _configure_data_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setVerticalSpacing(14)
        lo.setColumnStretch(0, 1)
        image = self._add_settings_group(frame, lo, 0, "Image preparation")
        cache = self._add_settings_group(frame, lo, 1, "Latent cache")
        self.build_data_tab_content(image, cache, self.controller, self.ui_state)
        lo.setRowStretch(2, 1)

    def _configure_backup_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setVerticalSpacing(14)
        lo.setColumnStretch(0, 1)
        backups = self._add_settings_group(frame, lo, 0, "Automatic backups")
        saves = self._add_settings_group(frame, lo, 1, "Model saves")
        self.build_backup_tab_content(backups, saves, self.controller, self.ui_state)
        lo.setRowStretch(2, 1)

    @staticmethod
    def _add_settings_group(frame, layout, row, title):
        group = QGroupBox(title, frame)
        grid = pyside6_components._layout(group)
        grid.setContentsMargins(16, 18, 16, 16)
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        layout.addWidget(group, row, 0)
        return group

    def _configure_tools_frame(self, frame):
        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setVerticalSpacing(14)
        lo.setColumnStretch(0, 1)
        media = self._add_settings_group(frame, lo, 0, "Dataset and media")
        diagnostics = self._add_settings_group(frame, lo, 1, "Model and diagnostics")
        self.build_tools_tab_content(media, diagnostics, self.controller, self.ui_state)
        lo.setRowStretch(2, 1)

    def _configure_embedding_frame(self, frame):
        self.build_embedding_tab_content(frame, self.controller, self.ui_state)
        pyside6_components._pack_form(frame)

    def _create_tabs(self):
        general_page = self._create_scrollable_tab(self._configure_general_frame)
        self.tabview.addTab(general_page, "general")
        self._tab_widgets["general"] = general_page

        self.model_tab = PySide6ModelTabView(None, ModelTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(self.model_tab, "model")
        self._tab_widgets["model"] = self.model_tab

        data_page = self._create_scrollable_tab(self._configure_data_frame)
        self.tabview.addTab(data_page, "data")
        self._tab_widgets["data"] = data_page

        concepts_page = QWidget()
        self.concepts_tab = PySide6ConceptTabView(concepts_page, ConceptTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(concepts_page, "concepts")
        self._tab_widgets["concepts"] = concepts_page

        self.training_tab = PySide6TrainingTabView(None, TrainingTabController(self.controller.train_config), self.ui_state)
        self.tabview.addTab(self.training_tab, "training")
        self._tab_widgets["training"] = self.training_tab

        sampling_page = self.create_sampling_tab()
        self.tabview.addTab(sampling_page, "sampling")
        self._tab_widgets["sampling"] = sampling_page

        backup_page = self._create_scrollable_tab(self._configure_backup_frame)
        self.tabview.addTab(backup_page, "backup")
        self._tab_widgets["backup"] = backup_page

        tools_page = self._create_scrollable_tab(self._configure_tools_frame)
        self.tabview.addTab(tools_page, "tools")
        self._tab_widgets["tools"] = tools_page

        additional_embeddings_page = QWidget()
        self.additional_embeddings_tab = PySide6AdditionalEmbeddingsTabView(
            additional_embeddings_page,
            AdditionalEmbeddingsTabController(self.controller.train_config),
            self.ui_state,
        )
        self.tabview.addTab(additional_embeddings_page, "additional embeddings")
        self._tab_widgets["additional embeddings"] = additional_embeddings_page

        self.cloud_tab = PySide6CloudTabView(None, CloudTabController(self.controller.train_config, self), self.ui_state)
        self.tabview.addTab(self.cloud_tab, "cloud")
        self._tab_widgets["cloud"] = self.cloud_tab

    def _refresh_navigation(self):
        if self.navigation is None or self.tabview is None:
            return
        available = {
            key for key, page in self._tab_widgets.items()
            if (index := self.tabview.indexOf(page)) >= 0 and self.tabview.isTabVisible(index)
        }
        self.navigation.set_pages(available)
        self._sync_navigation_selection()
        retranslate_tree(self.navigation)

    def _sync_navigation_selection(self, _index: int | None = None):
        if self.navigation is None or self.tabview is None:
            return
        current = self.tabview.currentWidget()
        key = next((key for key, page in self._tab_widgets.items() if page is current), None)
        self.navigation.select_page(key)
        title, subtitle = self.PAGE_COPY.get(key, ("OneTrainer", ""))
        self.page_heading.setText(tr(title))
        self.page_subtitle.setText(tr(subtitle))

    def _change_theme(self, theme: str):
        apply_theme(QApplication.instance(), theme, persist=True)
        self.navigation.set_theme(theme)

    def _change_language(self, language: str):
        set_language(QApplication.instance(), language, persist=True)
        self.navigation.set_language(language)
        self._sync_navigation_selection()
        self._fit_action_labels()

    def _show_navigation_page(self, key: str):
        page = self._tab_widgets.get(key)
        if page is None or self.tabview is None:
            return
        index = self.tabview.indexOf(page)
        if index >= 0 and self.tabview.isTabVisible(index):
            self.tabview.setCurrentIndex(index)

    def create_sampling_tab(self):
        tab_page = QWidget()
        tab_lo = QGridLayout(tab_page)
        tab_lo.setContentsMargins(0, 0, 0, 0)
        tab_lo.setSpacing(0)
        tab_lo.setRowStretch(0, 0)
        tab_lo.setRowStretch(1, 1)
        tab_lo.setColumnStretch(0, 1)

        top_frame = QWidget(tab_page)
        tab_lo.addWidget(top_frame, 0, 0)
        top_lo = pyside6_components._layout(top_frame)
        top_lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)

        sub_frame = QWidget(top_frame)
        top_lo.addWidget(sub_frame, 2, 0, 1, 4)

        self.build_sampling_tab_header(top_frame, sub_frame, self.controller, self.ui_state)
        top_lo.setColumnStretch(1, 1)
        top_lo.setColumnStretch(3, 1)
        pyside6_components._layout(sub_frame).setColumnStretch(4, 1)

        sampling_container = QWidget(tab_page)
        tab_lo.addWidget(sampling_container, 1, 0)
        self.sampling_tab = PySide6SamplingTabView(
            sampling_container, SamplingTabController(self.controller.train_config), self.ui_state
        )

        return tab_page

    def open_profiling_tool(self):
        self.profiling_window.show()

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()
        if self.training_tab:
            self.training_tab.refresh_ui()
        if self.lora_tab:
            self.lora_tab.refresh_ui()
        self._update_additional_embeddings_tab(model_type)

    def _update_additional_embeddings_tab(self, model_type: ModelType):
        # additional embeddings only apply to models that support embedding training
        supported = TrainingMethod.EMBEDDING in model_type.supported_training_methods()
        page = self._tab_widgets.get("additional embeddings")
        if page is not None:
            self.tabview.setTabVisible(self.tabview.indexOf(page), supported)
        self._refresh_navigation()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        if training_method != TrainingMethod.LORA and 'LoRA' in self._tab_widgets:
            self._remove_training_method_tab('LoRA')
            self.lora_tab = None
        if training_method != TrainingMethod.EMBEDDING and 'embedding' in self._tab_widgets:
            self._remove_training_method_tab('embedding')

        if training_method == TrainingMethod.LORA and 'LoRA' not in self._tab_widgets:
            self.lora_tab = PySide6LoraTabView(None, LoraTabController(self.controller.train_config), self.ui_state)
            self.tabview.addTab(self.lora_tab, 'LoRA')
            self._tab_widgets['LoRA'] = self.lora_tab
        if training_method == TrainingMethod.EMBEDDING and 'embedding' not in self._tab_widgets:
            tab_page = self._create_scrollable_tab(self._configure_embedding_frame)
            self.tabview.addTab(tab_page, 'embedding')
            self._tab_widgets['embedding'] = tab_page
        self._refresh_navigation()

    def _remove_training_method_tab(self, key: str):
        page = self._tab_widgets.pop(key)
        index = self.tabview.indexOf(page)
        if index >= 0:
            self.tabview.removeTab(index)
        # removeTab only detaches the page from QTabWidget. Destroy its fields
        # as well so their validators and UIState traces cannot outlive it.
        page.hide()
        page.deleteLater()

    def load_preset(self):
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()

    def _set_training_button_style(self, mode: str):
        if not self.training_button:
            return
        styles = {
            "idle":     ("Start Training", True, "startTrainingAction"),
            "running":  ("Stop Training", True, "stopTrainingAction"),
            "stopping": ("Stopping...", False, "stopTrainingAction"),
        }
        text, enabled, role = styles.get(mode, styles["idle"])
        self.training_button.setText(tr(text))
        self.training_button.setEnabled(enabled)
        if self.training_button.objectName() != role:
            self.training_button.setObjectName(role)
            self.training_button.style().unpolish(self.training_button)
            self.training_button.style().polish(self.training_button)
            self.training_button.update()

    def export_training(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, tr("Export Training Config"), "config.json",
            tr("JSON Files (*.json);;All Files (*.*)")
        )
        if file_path:
            errors = flush_and_validate_for_save(self.ui_state)
            if errors:
                QMessageBox.warning(self, tr("Cannot save configuration"), "\n".join(errors))
                return
            self.controller.export_training(file_path)

    def generate_debug_package(self):
        dir_path = QFileDialog.getExistingDirectory(self, tr("Select Directory to Save Debug Package"), ".")
        if not dir_path:
            return
        self.controller.generate_debug_package(Path(dir_path) / "OneTrainer_debug_report.zip")
