import unittest
from types import SimpleNamespace

from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin

import torch


class TokenizerWithoutCopy:
    def __deepcopy__(self, memo):
        raise AssertionError("Tokenizer must not be copied or reloaded")


class Pipeline:
    def __init__(self, tokenizer, fail=False):
        self.tokenizer = tokenizer
        self.other_tokenizer = tokenizer
        self.weights = torch.ones(2, dtype=torch.float32)
        self.fail = fail

    def to(self, *, device, dtype, silence_dtype_warnings):
        if self.fail:
            raise RuntimeError("Conversion failed")
        self.weights = self.weights.to(device=device, dtype=dtype)


class DtypeModelSaverTest(unittest.TestCase):
    def test_pipeline_weights_are_copied_and_tokenizers_are_shared(self):
        tokenizer = TokenizerWithoutCopy()
        pipeline = Pipeline(tokenizer)
        copied = DtypeModelSaverMixin()._copy_pipeline_to_dtype(pipeline, torch.float16, tokenizer, None, tokenizer)
        self.assertIsNot(copied, pipeline)
        self.assertIs(copied.tokenizer, tokenizer)
        self.assertIs(copied.other_tokenizer, tokenizer)
        self.assertEqual(copied.weights.dtype, torch.float16)
        self.assertEqual(pipeline.weights.dtype, torch.float32)
        copied.weights.zero_()
        self.assertTrue(torch.all(pipeline.weights == 1))
        self.assertEqual(vars(tokenizer), {})

    def test_failure_preserves_existing_tokenizer_copy_hook(self):
        def copy_hook(memo):
            raise AssertionError("Original tokenizer hook must not be invoked")

        tokenizer = SimpleNamespace(__deepcopy__=copy_hook)
        pipeline = Pipeline(tokenizer, fail=True)
        with self.assertRaisesRegex(RuntimeError, "Conversion failed"):
            DtypeModelSaverMixin()._copy_pipeline_to_dtype(pipeline, torch.float16, tokenizer)
        self.assertIs(tokenizer.__deepcopy__, copy_hook)
        self.assertEqual(pipeline.weights.dtype, torch.float32)

    def test_no_dtype_returns_original_pipeline(self):
        pipeline = Pipeline(TokenizerWithoutCopy())
        self.assertIs(DtypeModelSaverMixin()._copy_pipeline_to_dtype(pipeline, None), pipeline)


if __name__ == "__main__":
    unittest.main()
