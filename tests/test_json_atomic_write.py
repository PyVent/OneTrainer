import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from modules.util.path_util import write_json_atomic


class AtomicJsonWriteTest(unittest.TestCase):
    def test_success_replaces_the_complete_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"old": true}', encoding="utf-8")

            write_json_atomic(str(path), {"name": "Тест", "learning_rate": 0.0003})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "name": "Тест",
                    "learning_rate": 0.0003,
                },
            )
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_serialization_failure_preserves_old_file_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"old": true}', encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomic(str(path), {"invalid": object()})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_parallel_saves_never_share_a_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            barrier = threading.Barrier(2)
            original_dump = json.dump
            temporary_paths = []

            def synchronized_dump(obj, stream, **kwargs):
                temporary_paths.append(stream.name)
                barrier.wait(timeout=5)
                original_dump(obj, stream, **kwargs)

            with (
                patch("modules.util.path_util.json.dump", side_effect=synchronized_dump),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                results = [executor.submit(write_json_atomic, str(path), {"value": value}) for value in (1, 2)]
                for result in results:
                    result.result(timeout=10)

            self.assertEqual(len(set(temporary_paths)), 2)
            self.assertIn(json.loads(path.read_text(encoding="utf-8")), ({"value": 1}, {"value": 2}))
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
