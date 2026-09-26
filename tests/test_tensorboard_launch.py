import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.util.tensorboard_util import tensorboard_executable


class TensorBoardExecutableTest(unittest.TestCase):
    def test_windows_console_script_is_found_in_active_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "tensorboard.exe"
            executable.touch()
            with patch("modules.util.tensorboard_util.sysconfig.get_path", return_value=directory), \
                    patch("modules.util.tensorboard_util.sys.platform", "win32"):
                self.assertEqual(tensorboard_executable(), str(executable))

    def test_missing_console_script_has_actionable_error(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("modules.util.tensorboard_util.sysconfig.get_path", return_value=directory),
            patch("modules.util.tensorboard_util.sys.platform", "win32"),
            self.assertRaisesRegex(FileNotFoundError, "requirements-global.txt"),
        ):
            tensorboard_executable()


if __name__ == "__main__":
    unittest.main()
