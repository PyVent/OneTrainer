import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class RecordingProgress:
    instances = []

    def __init__(self, *, desc, unit):
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.total = None
        self.postfix = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, amount=1):
        self.n += amount

    def set_postfix(self, *, refresh, **values):
        self.postfix = values


class FakeCollectPaths:
    def __init__(
        self,
        concept_in_name,
        path_in_name,
        include_subdirectories_in_name,
        enabled_in_name,
        path_out_name,
        concept_out_name,
        extensions,
        include_postfix,
        exclude_postfix,
    ):
        self.concept_in_name = concept_in_name
        self.path_in_name = path_in_name
        self.include_subdirectories_in_name = include_subdirectories_in_name
        self.enabled_in_name = enabled_in_name
        self.path_out_name = path_out_name
        self.concept_out_name = concept_out_name
        self.extensions = [extension.lower() for extension in extensions]
        self.include_postfix = include_postfix
        self.exclude_postfix = exclude_postfix
        self.paths = []
        self.concepts = []


def load_collector():
    mgds = ModuleType("mgds")
    pipeline_modules = ModuleType("mgds.pipelineModules")
    collect_paths = ModuleType("mgds.pipelineModules.CollectPaths")
    collect_paths.CollectPaths = FakeCollectPaths
    tqdm = ModuleType("tqdm")
    tqdm.tqdm = RecordingProgress
    source = Path(__file__).resolve().parents[1] / "modules" / "dataLoader" / "ProgressCollectPaths.py"
    spec = importlib.util.spec_from_file_location("sample_path_progress_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        "sys.modules",
        {
            "mgds": mgds,
            "mgds.pipelineModules": pipeline_modules,
            "mgds.pipelineModules.CollectPaths": collect_paths,
            "tqdm": tqdm,
        },
    ):
        spec.loader.exec_module(module)
    return module.ProgressCollectPaths


class SamplePathProgressTest(unittest.TestCase):
    def setUp(self):
        RecordingProgress.instances.clear()

    def test_counts_checked_paths_and_preserves_sample_selection(self):
        collector_type = load_collector()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "samples"
            root.mkdir()
            (root / "z.jpg").touch()
            (root / "z-masklabel.png").touch()
            (root / "readme.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "a.PNG").touch()
            (nested / "a-condlabel.png").touch()
            hidden = root / ".cache"
            hidden.mkdir()
            (hidden / "ignored.jpg").touch()
            disabled = Path(directory) / "disabled"
            disabled.mkdir()
            (disabled / "unused.jpg").touch()

            concepts = [
                {"path": str(root), "enabled": True, "include_subdirectories": True},
                {"path": str(disabled), "enabled": False, "include_subdirectories": True},
            ]
            collector = collector_type(
                concept_in_name="concept",
                path_in_name="path",
                include_subdirectories_in_name="concept.include_subdirectories",
                enabled_in_name="enabled",
                path_out_name="image_path",
                concept_out_name="concept",
                extensions={".jpg", ".png"},
                include_postfix=None,
                exclude_postfix=["-masklabel", "-condlabel"],
            )
            collector._get_previous_length = lambda name: len(concepts)
            collector._get_previous_item = lambda variation, name, index: (
                concepts[index] if name == "concept" else concepts[index]["include_subdirectories"]
            )

            collector.start(0)

            self.assertEqual(collector.paths, [str(nested / "a.PNG"), str(root / "z.jpg")])
            self.assertEqual(collector.concepts, [concepts[0], concepts[0]])
            progress = RecordingProgress.instances[0]
            self.assertIsNone(progress.total)
            self.assertEqual(progress.unit, "path")
            self.assertEqual(progress.n, 7)
            self.assertEqual(progress.postfix, {"files": 5, "samples": 2})

    def test_non_recursive_and_include_postfix(self):
        collector_type = load_collector()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image-a.png").touch()
            (root / "image-b.png").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "nested-a.png").touch()
            concept = {"path": str(root), "enabled": True, "include_subdirectories": False}
            collector = collector_type(
                concept_in_name="concept",
                path_in_name="path",
                include_subdirectories_in_name="concept.include_subdirectories",
                enabled_in_name="enabled",
                path_out_name="image_path",
                concept_out_name="concept",
                extensions={".png"},
                include_postfix=["-a"],
                exclude_postfix=[],
            )
            collector._get_previous_length = lambda name: 1
            collector._get_previous_item = lambda variation, name, index: concept if name == "concept" else False

            collector.start(0)

            self.assertEqual(collector.paths, [str(root / "image-a.png")])
            progress = RecordingProgress.instances[0]
            self.assertEqual(progress.n, 3)
            self.assertEqual(progress.postfix, {"files": 2, "samples": 1})


if __name__ == "__main__":
    unittest.main()
