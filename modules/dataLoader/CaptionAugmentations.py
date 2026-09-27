"""Keep MGDS tag augmentation semantics, while leaving prose captions intact."""

from mgds.pipelineModules.CapitalizeTags import CapitalizeTags as MGDSCapitalizeTags
from mgds.pipelineModules.DropTags import DropTags as MGDSDropTags
from mgds.pipelineModules.ShuffleTags import ShuffleTags as MGDSShuffleTags


class _TagsOnly:
    def get_inputs(self):
        return super().get_inputs() + ["caption_is_tags"]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        if not self._get_previous_item(variation, "caption_is_tags", index):
            return {self.text_out_name: self._get_previous_item(variation, self.text_in_name, index)}
        return super().get_item(variation, index, requested_name)


class DropTags(_TagsOnly, MGDSDropTags):
    pass


class CapitalizeTags(_TagsOnly, MGDSCapitalizeTags):
    pass


class ShuffleTags(_TagsOnly, MGDSShuffleTags):
    pass
