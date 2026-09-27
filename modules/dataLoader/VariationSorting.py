from modules.dataLoader.cache_grouping import init_concept_groups

from mgds.pipelineModules.VariationSorting import VariationSorting as MGDSVariationSorting


class VariationSorting(MGDSVariationSorting):
    """Advance uncached augmentations for each repetition, including within an epoch."""

    def get_item(self, variation: int, index: int, requested_name: str = None):
        offset = 0
        for group, count in self.group_output_samples.items():
            if index < offset + count:
                indices = self.group_indices[group]
                local_index = index - offset + variation * count
                in_variation, group_index = divmod(local_index, len(indices))
                return {requested_name: self._get_previous_item(in_variation, requested_name, indices[group_index])}
            offset += count
        raise IndexError(index)

    def start(self, variation):
        if not self.variations_initialized:
            init_concept_groups(self, cached=False)
