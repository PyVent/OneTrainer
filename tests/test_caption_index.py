import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.util.caption_index import CaptionCountIndex
from modules.util.caption_util import read_caption_lines


class CaptionIndexTest(unittest.TestCase):
    def test_reuses_counts_and_reads_only_changed_or_added_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, missing = (str(root / name) for name in ("a.txt", "b.txt", "missing.txt"))
            Path(first).write_text("\ufefftags, words, colors\n\nA description.\n", encoding="utf-8")
            Path(second).write_text("One.\n", encoding="utf-8")

            def sources():
                return {path: (os.stat(path).st_size, os.stat(path).st_mtime_ns) if os.path.exists(path) else (None, None)
                        for path in (first, second, missing)}

            cache = CaptionCountIndex(root, "concept")
            self.assertEqual(cache.counts(sources()), {first: 2, second: 1, missing: 1})
            with patch("modules.util.caption_index.read_caption_lines", side_effect=AssertionError("Unchanged TXT was opened")):
                self.assertEqual(CaptionCountIndex(root, "concept").counts(sources()), {first: 2, second: 1, missing: 1})

            Path(second).write_text("One.\nTwo.\nThree.\n", encoding="utf-8")
            Path(missing).write_text("new\ncaption", encoding="utf-8")
            with patch("modules.util.caption_index.read_caption_lines", wraps=read_caption_lines) as read:
                self.assertEqual(CaptionCountIndex(root, "concept").counts(sources()), {first: 2, second: 3, missing: 2})
                self.assertEqual({call.args[0] for call in read.call_args_list}, {second, missing})
            Path(second).unlink()
            self.assertEqual(CaptionCountIndex(root, "concept").counts(sources()), {first: 2, second: 1, missing: 2})

    def test_corrupt_index_is_rebuilt_and_invalid_utf8_still_reports_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = str(root / "caption.txt")
            Path(source).write_text("first\nsecond", encoding="utf-8")
            stamp = (12, 1234)
            index = CaptionCountIndex(root, "concept")
            index.path.parent.mkdir(parents=True)
            index.path.write_text("interrupted JSON", encoding="utf-8")
            self.assertEqual(CaptionCountIndex(root, "concept").counts({source: stamp}), {source: 2})
            Path(source).write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(ValueError, "Caption file must be UTF-8"):
                CaptionCountIndex(root, "concept").counts({source: (2, 5678)})


if __name__ == "__main__":
    unittest.main()
