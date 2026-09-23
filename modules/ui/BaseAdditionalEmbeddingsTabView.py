
from modules.ui.BaseConfigListView import BaseConfigListView


class BaseAdditionalEmbeddingsTabView(BaseConfigListView):

    def refresh_ui(self):
        if self.element_list is not None:
            self._destroy_frame(self.element_list)
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

    def open_element_window(self, i, ui_state):
        pass
