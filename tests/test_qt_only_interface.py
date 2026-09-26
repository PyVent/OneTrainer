"""Keep the supported UI entrypoints and production modules on Qt."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_GUI_PACKAGES = {"tkinter", "customtkinter"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class QtOnlyInterfaceTest(unittest.TestCase):
    def test_production_python_has_no_legacy_gui_imports(self):
        offenders = []
        for folder in (ROOT / "modules", ROOT / "scripts"):
            for path in folder.rglob("*.py"):
                offenders.extend(
                    f"{path.relative_to(ROOT)}: {module}"
                    for module in imported_modules(path)
                    if module.split(".", 1)[0] in LEGACY_GUI_PACKAGES
                )
        self.assertEqual(offenders, [])

    def test_entrypoints_launch_qt(self):
        wrapper_imports = imported_modules(ROOT / "scripts/train_ui.py")
        self.assertIn("train_ui_qt", wrapper_imports)

        qt_entrypoints = {
            "train_ui_qt.py": "modules.ui.PySide6TrainUIView",
            "caption_ui.py": "modules.ui.PySide6CaptionUIView",
            "convert_model_ui.py": "modules.ui.PySide6ConvertModelUIView",
            "video_tool_ui.py": "modules.ui.PySide6VideoToolUIView",
        }
        for filename, view_module in qt_entrypoints.items():
            with self.subTest(filename=filename):
                imports = imported_modules(ROOT / "scripts" / filename)
                self.assertIn(view_module, imports)
                self.assertIn("modules.util.ui.pyside6_util", imports)

        self.assertIn("train_ui_qt.py", (ROOT / "start-ui.bat").read_text(encoding="utf-8"))
        self.assertIn("start-ui.bat", (ROOT / "start.bat").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
