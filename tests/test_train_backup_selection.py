import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.backup_util import COMPLETE_MARKER
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress


def write_model_backup(path):
    path = Path(path)
    (path / "optimizer").mkdir(parents=True)
    (path / "optimizer" / "optimizer.pt").write_bytes(b"optimizer")
    (path / "lora").mkdir()
    (path / "lora" / "lora.safetensors").write_bytes(b"weights")
    (path / "meta.json").write_text(json.dumps({
        "train_progress": {"epoch": 1, "epoch_step": 2, "epoch_sample": 3, "global_step": 4},
    }), encoding="utf-8")


class TrainBackupSelectionTest(unittest.TestCase):
    def test_published_backup_resumes_and_pruning_ignores_incomplete_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig.default_values()
            config.workspace_dir = directory
            config.rolling_backup = True
            config.rolling_backup_count = 1
            root = Path(directory) / "backup"
            old = root / "2026-01-01-backup-0-0-0"
            old.mkdir(parents=True)
            write_model_backup(old)
            (old / "onetrainer_config").mkdir()
            (old / "onetrainer_config" / "args.json").write_text("{}", encoding="utf-8")
            (root / "2026-03-01-backup-empty").mkdir()

            trainer = object.__new__(GenericTrainer)
            trainer.config = config
            trainer.callbacks = SimpleNamespace(on_update_status=Mock())
            trainer.model = object()
            trainer.model_setup = SimpleNamespace(setup_train_device=Mock())

            trainer.model_saver = SimpleNamespace(save=Mock(side_effect=OSError("disk full")))
            with (
                patch("modules.trainer.GenericTrainer.get_string_timestamp", return_value="2026-01-15_00-00-00"),
                patch("modules.trainer.GenericTrainer.torch_gc"),
                patch("modules.trainer.GenericTrainer.traceback.print_exc"),
                patch("modules.trainer.GenericTrainer.tqdm.write"),
            ):
                trainer._GenericTrainer__backup(TrainProgress(), print_msg=False)
            self.assertTrue(old.exists())
            self.assertFalse((root / "2026-01-15_00-00-00-backup-0-0-0-v2").exists())

            trainer.model_saver = SimpleNamespace(save=lambda _model, _type, _format, path, _dtype: write_model_backup(path))
            with (
                patch("modules.trainer.GenericTrainer.get_string_timestamp", return_value="2026-02-01_00-00-00"),
                patch("modules.trainer.GenericTrainer.torch_gc"),
            ):
                trainer._GenericTrainer__backup(TrainProgress(), print_msg=False)

            published = root / "2026-02-01_00-00-00-backup-0-0-0-v2"
            self.assertTrue((published / COMPLETE_MARKER).is_file())
            self.assertEqual(config.get_last_backup_path(), str(published))
            self.assertFalse(old.exists())
            self.assertTrue((root / "2026-03-01-backup-empty").exists())


if __name__ == "__main__":
    unittest.main()
