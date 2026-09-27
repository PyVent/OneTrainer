import tempfile
import unittest
from pathlib import Path

from modules.util.caption_util import (
    caption_format,
    caption_key,
    caption_sample_count,
    load_captions,
    read_caption_lines,
)
from modules.util.config.ConceptConfig import ConceptConfig


class CaptionRulesTest(unittest.TestCase):
    def test_legacy_configs_keep_random_selection_and_tag_processing(self):
        for version in (0, 1, 2):
            config = ConceptConfig.default_values()
            config.text.caption_mode, config.text.caption_format = "all", "auto"
            config.from_dict({"__version": version, "text": {"enable_tag_shuffling": True}})
            self.assertEqual(config.text.caption_mode, "random")
            self.assertEqual(config.text.caption_format, "tags")

    def test_mixed_format_and_content_overrides_survive_reordering_and_save(self):
        config = ConceptConfig.default_values()
        config.text.caption_format = "auto"
        self.assertEqual(caption_format("red, blue", config.text.to_dict()), "text")
        self.assertEqual(caption_format("red, blue, green", config.text.to_dict()), "tags")
        self.assertEqual(caption_format("a red car, a blue sky, a green tree", config.text.to_dict()), "text")
        config.text.caption_overrides[caption_key("red, blue")] = "tags"
        saved = ConceptConfig.default_values().from_dict(config.to_dict())
        self.assertEqual(caption_format("red, blue", saved.text.to_dict()), "tags")
        self.assertEqual(caption_format("Unrelated caption", saved.text.to_dict()), "text")

    def test_sources_bom_blank_lines_missing_files_and_encoding_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            caption = root / "sample.txt"
            self.assertEqual(load_captions(image, {}), [""])
            caption.write_text("\ufeff теги, фон, объект\n\nA description.\n", encoding="utf-8")
            lines = load_captions(image, {})
            self.assertEqual(lines, ["теги, фон, объект", "A description."])
            self.assertEqual(load_captions(image, {"prompt_source": "filename"}), ["sample"])
            self.assertEqual(load_captions(image, {"prompt_source": "concept", "prompt_path": str(caption)}), lines)
            self.assertEqual(caption_sample_count(lines, {"caption_mode": "all"}), 2)
            self.assertEqual(caption_sample_count(lines, {}), 1)
            caption.write_text(" \n\n", encoding="utf-8")
            self.assertEqual(read_caption_lines(caption), [""])
            caption.write_bytes(b"\xff\xfe\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8.*sample.txt"):
                read_caption_lines(caption)


if __name__ == "__main__":
    unittest.main()
