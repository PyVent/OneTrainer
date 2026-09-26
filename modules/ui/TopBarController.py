import json
import os
import traceback
import webbrowser
from contextlib import suppress

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.model_catalog import MODEL_TYPE_CHOICES
from modules.util.path_util import write_json_atomic


class TopBarController:
    def __init__(self, config: TrainConfig):
        self.train_config = config
        self.last_load_error: str | None = None

    def get_model_types(self) -> list[tuple[str, ModelType]]:
        return list(MODEL_TYPE_CHOICES)

    def get_training_methods(self, model_type: ModelType) -> list[tuple[str, TrainingMethod]]:
        labels = {
            TrainingMethod.FINE_TUNE: "Fine Tune",
            TrainingMethod.LORA: "LoRA",
            TrainingMethod.EMBEDDING: "Embedding",
            TrainingMethod.FINE_TUNE_VAE: "Fine Tune VAE",
        }
        return [(labels[m], m) for m in model_type.supported_training_methods()]

    def load_preset_tree(self, dir: str = "training_presets") -> list[tuple[str, str | list]]:
        # mirrors whatever directory structure happens to exist under `dir`; a node is either
        # (display_name, path) for a leaf preset or (display_name, children) for a group.
        # "#" marks a built-in preset (vs. a user file); "#.json" is the last-session state,
        # not a preset. Same convention as load_config_from_file's is_built_in_preset check.
        nodes = []
        if os.path.isdir(dir):
            for entry in sorted(os.scandir(dir), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    children = self.load_preset_tree(entry.path)
                    if children:
                        nodes.append((entry.name, children))
                elif entry.name.startswith("#") and entry.name != "#.json" and entry.name.endswith(".json"):
                    nodes.append((os.path.splitext(entry.name)[0], path_util.canonical_join(dir, entry.name)))
        return nodes

    def save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))
        return path

    def save_config_to_path(self, path: str) -> None:
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))

    def load_config_from_file(self, filename: str) -> TrainConfig | None:
        self.last_load_error = None
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                # built-in configs are always saved in the most recent version, so migration can be skipped
                loaded_config = default_config.from_dict(
                    loaded_dict, migrate=not is_built_in_preset, strict=True,
                ).to_unpacked_config()

            with suppress(FileNotFoundError), open("secrets.json", "r") as f:
                secrets_dict = json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict, strict=True)

            self.train_config.from_dict(loaded_config.to_dict(), strict=True)
            return self.train_config
        except FileNotFoundError:
            self.last_load_error = f"File not found: {filename}"
            return None
        except Exception as exc:
            self.last_load_error = str(exc) or type(exc).__name__
            print(traceback.format_exc())
            return None

    def save_secrets(self, path) -> str:
        write_json_atomic(path, self.train_config.secrets.to_dict())
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def save_default(self):
        self.save_to_file("#")
        self.save_secrets("secrets.json")
