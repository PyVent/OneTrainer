import threading
import time

from modules.dataLoader.cache_grouping import init_concept_groups
from modules.util.loading_progress import loading_callback

from mgds.pipelineModules.DiskCache import DiskCache as MGDSDiskCache


class DiskCache(MGDSDiskCache):
    def _DiskCache__init_variations(self):
        init_concept_groups(self, cached=True)

    def start(self, out_variation):
        self._encoded_count = 0
        self._progress_lock = threading.Lock()
        self._last_progress = time.monotonic()
        self._report_status = loading_callback()
        self._cache_label = "image cache" if self.aggregate_names else "text cache"
        self._report_status(f"Preparing {self._cache_label}…")
        super().start(out_variation)
        self._report_status(f"Ready: {self._cache_label}")

    def _get_previous_item(self, variation, name, index):
        value = super()._get_previous_item(variation, name, index)
        final_name = (self.aggregate_names or self.split_names)[-1]
        if name == final_name:
            with self._progress_lock:
                self._encoded_count += 1
                if time.monotonic() - self._last_progress >= 0.25:
                    self._last_progress = time.monotonic()
                    self._report_status(f"{self._cache_label}: {self._encoded_count} examples encoded")
        return value
