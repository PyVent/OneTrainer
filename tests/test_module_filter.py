import unittest
from types import SimpleNamespace

from modules.util.ModuleFilter import ModuleFilter


class ModuleFilterTest(unittest.TestCase):
    def test_substring_filter_records_only_successful_matches(self):
        layer_filter = ModuleFilter(" attn ")
        self.assertFalse(layer_filter.was_used())
        self.assertFalse(layer_filter.matches("transformer.ff.proj"))
        self.assertFalse(layer_filter.was_used())
        self.assertTrue(layer_filter.matches("transformer.attn.to_v"))
        self.assertTrue(layer_filter.was_used())

    def test_regex_filter_and_invalid_pattern(self):
        layer_filter = ModuleFilter(r"attn[12]\.to_v$", use_regex=True)
        self.assertTrue(layer_filter.matches("transformer.attn2.to_v"))
        self.assertFalse(layer_filter.matches("transformer.attn2.to_k"))
        with self.assertRaisesRegex(ValueError, "Invalid regex pattern"):
            ModuleFilter("(", use_regex=True)

    def test_non_printable_pattern_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-printable"):
            ModuleFilter("attn\x00")

    def test_blank_entries_do_not_expand_a_specific_layer_selection(self):
        config = SimpleNamespace(layer_filter="attn, ,conv,", layer_filter_regex=False)
        filters = ModuleFilter.create(config)

        self.assertEqual(len(filters), 2)
        self.assertTrue(any(layer_filter.matches("transformer.attn.to_v") for layer_filter in filters))
        self.assertTrue(any(layer_filter.matches("transformer.conv.weight") for layer_filter in filters))
        self.assertFalse(any(layer_filter.matches("transformer.ff.proj") for layer_filter in filters))

    def test_completely_empty_selection_still_matches_all_layers(self):
        config = SimpleNamespace(layer_filter=" , ", layer_filter_regex=True)
        filters = ModuleFilter.create(config)

        self.assertEqual(len(filters), 1)
        self.assertTrue(filters[0].matches("any.layer"))


if __name__ == "__main__":
    unittest.main()
