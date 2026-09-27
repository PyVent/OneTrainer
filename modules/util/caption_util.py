"""Caption parsing and format rules shared by training, preview and dataset checks."""

import hashlib
import re
from pathlib import Path


def read_caption_lines(path: str | Path) -> list[str]:
    """Missing/empty captions keep one unconditional example, as in the legacy loader."""
    try:
        with open(path, encoding="utf-8-sig") as file:
            return [line.strip() for line in file if line.strip()] or [""]
    except FileNotFoundError:
        return [""]
    except UnicodeError as exc:
        raise ValueError(f"Caption file must be UTF-8: {path}") from exc


def load_captions(image_path: str | Path, settings: dict) -> list[str]:
    path = Path(image_path)
    match settings.get("prompt_source", "sample"):
        case "filename":
            return [path.stem]
        case "concept":
            source = settings.get("prompt_path", "")
            return read_caption_lines(source) if source else [""]
        case _:
            return read_caption_lines(path.with_suffix(".txt"))


def caption_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def caption_format(text: str, settings: dict) -> str:
    """Conservative heuristic; content-based overrides survive reordered caption lines.

    Auto is deliberately opt-in. It cannot identify every natural-language caption;
    uncertain strings are left intact, and the editor exposes the result and override.
    """
    override = settings.get("caption_overrides", {}).get(caption_key(text))
    if override in ("tags", "text"):
        return override
    mode = settings.get("caption_format", "tags")
    if mode in ("tags", "text"):
        return mode
    if mode != "auto":
        raise ValueError(f"Unknown caption format: {mode}")
    delimiter = settings.get("tag_delimiter", ",")
    parts = [part.strip() for part in text.split(delimiter) if part.strip()] if delimiter else []
    # Long clauses, sentence punctuation and complete sentences are prose. A list
    # of at least three short fragments is a candidate for tag augmentation.
    if len(parts) < 3 or re.search(r"[.!?。！？](?:\s|$)", text):
        return "text"
    words = [len(part.replace("_", " ").split()) for part in parts]
    sentence_start = re.compile(r"^(?:a|an|the|this|that|there|it|he|she|they|we|i)\s", re.IGNORECASE)
    if max(words) > 5 or sum(words) / len(words) > 3 or any(sentence_start.match(p) for p in parts):
        return "text"
    return "tags"


def caption_sample_count(captions: list[str], settings: dict) -> int:
    mode = settings.get("caption_mode", "random")
    if mode not in ("random", "all"):
        raise ValueError(f"Unknown caption selection mode: {mode}")
    return len(captions) if mode == "all" else 1
