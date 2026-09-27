
from modules.ui.ConceptWindowController import ConceptWindowController
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig


class ConceptTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def create_new_element(self) -> ConceptConfig:
        concept = ConceptConfig.default_values()
        concept.text.caption_mode = "all"
        concept.text.caption_format = "auto"
        return concept

    def randomize_seed(self, concept: ConceptConfig) -> ConceptConfig:
        concept.seed = ConceptConfig.default_values().seed
        return concept

    def open_element_window(self, parent, concept_config, ui_state, image_ui_state, text_ui_state, view_cls):
        return view_cls(parent, ConceptWindowController(self.train_config, concept_config), ui_state, image_ui_state, text_ui_state)
