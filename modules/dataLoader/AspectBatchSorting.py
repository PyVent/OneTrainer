from modules.util.loading_progress import LoadingProgress, report_loading_status

from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting as MGDSAspectBatchSorting
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting as MGDSInlineAspectBatchSorting


class AspectBatchSorting(MGDSAspectBatchSorting):
    def _AspectBatchSorting__sort_resolutions(self, variation):
        self.bucket_dict = {}
        total = self._get_previous_length(self.resolution_in_name)
        with LoadingProgress(total=total, desc="sorting aspect buckets", unit="example") as progress:
            for index in range(total):
                resolution = self._get_previous_item(self.current_variation, self.resolution_in_name, index)
                self.bucket_dict.setdefault(resolution, []).append(index)
                progress.update()


class InlineAspectBatchSorting(MGDSInlineAspectBatchSorting):
    def start(self, variation, start_index):
        self._preparing = True
        self._prepared_count = 0
        report_loading_status("Preparing the first image batch…")
        try:
            super().start(variation, start_index)
        finally:
            self._preparing = False

    def _get_previous_item(self, variation, name, index):
        value = super()._get_previous_item(variation, name, index)
        if self._preparing and name == self.names[-1]:
            self._prepared_count += 1
            report_loading_status(f"Preparing the first image batch: {self._prepared_count} examples")
        return value
