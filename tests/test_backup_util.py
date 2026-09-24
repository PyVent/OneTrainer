import json
import tempfile
import unittest
from pathlib import Path

from modules.util.backup_util import COMPLETE_MARKER, complete_backup_paths, is_complete_backup, save_backup


def write_resume_artifacts(path: str | Path) -> None:
    path = Path(path)
    (path / "optimizer").mkdir(parents=True)
    (path / "optimizer" / "optimizer.pt").write_bytes(b"optimizer state")
    (path / "lora").mkdir()
    (path / "lora" / "lora.safetensors").write_bytes(b"model weights")
    (path / "meta.json").write_text(json.dumps({
        "train_progress": {"epoch": 1, "epoch_step": 2, "epoch_sample": 3, "global_step": 4},
    }), encoding="utf-8")
    (path / "onetrainer_config").mkdir()
    (path / "onetrainer_config" / "args.json").write_text("{}", encoding="utf-8")


class BackupUtilTest(unittest.TestCase):
    def test_only_complete_backup_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "2026-01-01-backup"
            legacy.mkdir()
            write_resume_artifacts(legacy)
            (root / "2026-01-03-backup").mkdir()  # interrupted before any data was written
            staged = root / ".incomplete-2026-01-04-backup"
            staged.mkdir()
            write_resume_artifacts(staged)

            self.assertEqual(complete_backup_paths(root), [str(legacy)])

            newest = Path(save_backup(root, "2026-01-05-backup-v2", write_resume_artifacts))
            self.assertTrue((newest / COMPLETE_MARKER).is_file())
            self.assertEqual(complete_backup_paths(root), [str(newest), str(legacy)])

            (newest / "lora" / "lora.safetensors").write_bytes(b"truncated")
            self.assertFalse(is_complete_backup(newest))
            self.assertEqual(complete_backup_paths(root), [str(legacy)])

            (newest / COMPLETE_MARKER).unlink()
            self.assertFalse(is_complete_backup(newest))

    def test_failed_write_leaves_no_visible_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def interrupted(path):
                Path(path, "partial.safetensors").write_bytes(b"partial")
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                save_backup(root, "2026-01-02-backup-v2", interrupted)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
