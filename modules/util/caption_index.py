"""Reusable caption counts, invalidated by the file metadata from directory scanning."""

import hashlib
import json
import os
import tempfile
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modules.util.caption_util import read_caption_lines
from modules.util.loading_progress import LoadingProgress


class CaptionCountIndex:
    def __init__(self, cache_dir, identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path = Path(cache_dir) / "caption_index" / (key + ".json") if cache_dir else None
        self.entries = {}
        if self.path is not None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("version") == 1 and isinstance(data.get("entries"), dict):
                    self.entries = data["entries"]
            except (OSError, ValueError, AttributeError):
                pass  # A missing or interrupted index is rebuilt from the TXT files.

    def counts(self, sources):
        """Return counts without retaining captions or scheduling an unbounded queue."""
        entries, pending = {}, []
        for source, stamp in sources.items():
            previous = self.entries.get(source)
            if stamp == (None, None):
                entries[source] = [*stamp, 1]
            elif isinstance(previous, list) and len(previous) == 3 and previous[:2] == list(stamp) and isinstance(previous[2], int) and previous[2] >= 1:
                entries[source] = previous
            else:
                pending.append(source)
        if pending:
            with (
                ThreadPoolExecutor(max_workers=8, thread_name_prefix="caption-count") as executor,
                LoadingProgress(total=len(pending), desc="reading changed caption files", unit="file") as progress,
            ):
                queue = deque()
                iterator = iter(pending)

                def submit():
                    source = next(iterator, None)
                    if source is not None:
                        queue.append((source, executor.submit(self._count, source)))

                for _ in range(min(32, len(pending))):
                    submit()
                while queue:
                    source, future = queue.popleft()
                    entries[source] = [*sources[source], future.result()]
                    progress.update()
                    submit()
        if entries != self.entries:
            self._save(entries)
        self.entries = entries
        return {source: value[2] for source, value in entries.items()}

    @staticmethod
    def _count(source):
        return len(read_caption_lines(source))

    def _save(self, entries):
        if self.path is None:
            return
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"version": 1, "entries": entries}, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, self.path)
        except OSError as error:
            warnings.warn(f"Could not save caption count index {self.path}: {error}", RuntimeWarning, stacklevel=2)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
