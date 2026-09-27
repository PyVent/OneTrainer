import hashlib
import json
import os
from array import array
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from modules.util.caption_index import CaptionCountIndex
from modules.util.caption_util import caption_format, caption_sample_count, read_caption_lines
from modules.util.loading_progress import LoadingProgress, report_loading_status

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ExpandCaptionSamples(PipelineModule, RandomAccessPipelineModule):
    """Index caption examples without retaining the entire dataset's text in RAM."""

    def __init__(self, use_collected_stats=False, cache_dir=None):
        super().__init__()
        self.use_collected_stats = use_collected_stats
        self.cache_dir = cache_dir
        self.images = []
        self.ends = array("Q")
        self._read_captions = lru_cache(maxsize=128)(read_caption_lines)

    def get_inputs(self):
        return ["image_path", "concept"] + (["sample_file_stats"] if self.use_collected_stats else [])

    def get_outputs(self):
        return ["image_path", "concept", "prompt", "caption_index", "caption_is_tags"]

    def length(self):
        return self.ends[-1] if self.ends else 0

    @staticmethod
    def _file_stamp(path):
        try:
            stat = os.stat(path)
            return stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return None, None

    @staticmethod
    def _hash(hasher, value):
        hasher.update(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        hasher.update(b"\n")

    def start(self, variation: int):
        groups = OrderedDict()
        count = self._get_previous_length("image_path")
        with LoadingProgress(total=count, desc="grouping caption inputs", unit="image") as progress:
            for index in range(count):
                concept = self._get_previous_item(0, "concept", index)
                groups.setdefault(id(concept), []).append(index)
                progress.update()

        self.images = []
        self.ends = array("Q")
        with LoadingProgress(total=count, desc="indexing caption examples", unit="image") as progress:
            for group_number, indices in enumerate(groups.values()):
                concept = dict(self._get_previous_item(0, "concept", indices[0]))
                settings = concept["text"]
                image_hash, text_hash = hashlib.sha256(), hashlib.sha256()
                self._hash(image_hash, ["caption-samples-v2", group_number, self.length()])
                source_kind = settings.get("prompt_source", "sample")
                shared_source = settings.get("prompt_path", "") if source_kind == "concept" else None
                shared_stamp = self._file_stamp(shared_source) if shared_source else (None, None)
                caption_sample_count([""], settings)  # Validate the mode even without a TXT file.
                counts = {}
                if settings.get("caption_mode", "random") == "all" and source_kind != "filename":
                    report_loading_status("Checking the caption count index…")
                    sources = {}
                    if source_kind == "concept":
                        if shared_source:
                            sources[shared_source] = shared_stamp
                    else:
                        for index in indices:
                            path = self._get_previous_item(0, "image_path", index)
                            source = str(Path(path).with_suffix(".txt"))
                            sources[source] = self._get_previous_item(0, "sample_file_stats", index)[1] if self.use_collected_stats else self._file_stamp(source)
                    cache = CaptionCountIndex(self.cache_dir, [os.path.abspath(concept["path"]), source_kind, shared_source, concept.get("include_subdirectories", False)])
                    counts = cache.counts(sources)
                for index in indices:
                    path = self._get_previous_item(0, "image_path", index)
                    source = shared_source if source_kind == "concept" else str(Path(path).with_suffix(".txt"))
                    if source_kind == "filename":
                        source = None
                    if self.use_collected_stats:
                        image_stamp, caption_stamp, mask_stamp, cond_stamp = self._get_previous_item(0, "sample_file_stats", index)
                    else:
                        stem = os.path.splitext(path)[0]
                        image_stamp, caption_stamp, mask_stamp, cond_stamp = [self._file_stamp(file) for file in (
                            path, stem + ".txt", stem + "-masklabel.png", stem + "-condlabel.png",
                        )]
                    examples = counts.get(source, 1)
                    self.images.append((path, concept, source, examples))
                    self.ends.append(self.length() + examples)
                    self._hash(image_hash, [path, image_stamp, mask_stamp, cond_stamp, examples])
                    self._hash(text_hash, [source, shared_stamp if source_kind == "concept" else caption_stamp])
                    progress.update()
                    if progress.n % 256 == 0:
                        progress.set_postfix(examples=self.length(), refresh=False)
                concept["_image_sample_key"] = image_hash.hexdigest()
                self._hash(text_hash, concept["_image_sample_key"])
                concept["_text_sample_key"] = text_hash.hexdigest()
                progress.set_postfix(examples=self.length(), refresh=False)

    def get_item(self, variation: int, index: int, requested_name: str = None):
        image_index = bisect_right(self.ends, index)
        path, concept, source, _count = self.images[image_index]
        if requested_name in ("concept", "image_path"):
            return {"image_path": path, "concept": concept}
        if source is None:
            captions = [Path(path).stem]
        elif source:
            captions = self._read_captions(source)
        else:
            captions = [""]
        offset = self.ends[image_index - 1] if image_index else 0
        caption_index = index - offset if concept["text"].get("caption_mode", "random") == "all" else self._get_rand(variation, index).randrange(len(captions))
        if caption_index >= len(captions):
            raise ValueError(f"Caption lines changed after dataset indexing: {source}. Restart training to rebuild the sample mapping.")
        prompt = captions[caption_index]
        return {"image_path": path, "concept": concept, "prompt": prompt, "caption_index": caption_index,
                "caption_is_tags": caption_format(prompt, concept["text"]) == "tags"}
