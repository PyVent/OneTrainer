import copy
import unittest
from unittest.mock import patch

from modules.util.optimizer.riemannion import init_manifold_
from modules.util.optimizer.riemannion_fast import _chol_inv_lt, _svd_small
from modules.util.optimizer.riemannion_util import RiemannionOT

import torch
from torch import nn


class RiemannionStabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_failed_finite_cholesky_factor_uses_column_normalization(self):
        gram = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[4.0, 0.0], [0.0, 9.0]]])
        factor, info = torch.linalg.cholesky_ex(gram)
        self.assertNotEqual(info[0].item(), 0)
        self.assertTrue(torch.isfinite(factor[0]).all())
        lower, inverse_transpose, keep = _chol_inv_lt(gram)
        expected = torch.diag_embed(torch.tensor([[1.000001, 1.000001]]).sqrt())[0]
        torch.testing.assert_close(lower[0], expected)
        self.assertTrue(keep.all())
        torch.testing.assert_close(lower @ inverse_transpose.mT, torch.eye(2).expand_as(lower))
        torch.testing.assert_close(lower[1] @ lower[1].T, gram[1], rtol=2e-6, atol=1e-5)

    def test_zero_and_rank_deficient_gram_matrices_stay_finite(self):
        gram = torch.stack([torch.zeros(4, 4), torch.ones(4, 4), torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0]))])
        lower, inverse_transpose, keep = _chol_inv_lt(gram)
        self.assertTrue(torch.isfinite(lower).all())
        self.assertTrue(torch.isfinite(inverse_transpose).all())
        self.assertFalse(keep[0].any())
        self.assertEqual(keep[2].tolist(), [True, False, False, False])

    def test_svd_reconstructs_repeated_zero_and_extreme_singular_values(self):
        matrices = torch.stack([
            torch.zeros(4, 4), torch.eye(4),
            torch.diag(torch.tensor([1e30, 1e30, 1e15, 0.0])),
            torch.diag(torch.tensor([1e-30, 1e-30, 0.0, 0.0])),
        ])
        left, singular, right = _svd_small(matrices)
        self.assertTrue(all(torch.isfinite(value).all() for value in (left, singular, right)))
        torch.testing.assert_close((left * singular.unsqueeze(-2)) @ right, matrices, rtol=2e-6, atol=1e-35)

    def test_failed_svd_recovers_with_float64_gesvd_and_preserves_shape(self):
        generator = torch.Generator().manual_seed(1)
        matrices = torch.randn(2, 3, 4, 4, generator=generator)
        matrices[0, 0] = 0
        with patch("torch.linalg.svd", side_effect=torch.linalg.LinAlgError("SVD did not converge")):
            left, singular, right = _svd_small(matrices)
        torch.testing.assert_close((left * singular.unsqueeze(-2)) @ right, matrices, rtol=2e-5, atol=1e-6)
        self.assertEqual(left.shape, matrices.shape)
        self.assertEqual(left.dtype, matrices.dtype)
        self.assertEqual(left.device, matrices.device)

    def test_nonfinite_svd_input_never_reaches_lapack(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), patch("torch.linalg.svd") as svd:
                with self.assertRaisesRegex(FloatingPointError, "NaN or Inf"):
                    _svd_small(torch.full((2, 4, 4), value))
                svd.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the batched optimizer path")
    def test_cuda_rank32_training_and_resume_with_svd_recovery(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                pairs = [(nn.Parameter(torch.randn(32, 80, device="cuda", dtype=dtype)),
                          nn.Parameter(torch.empty(96, 32, device="cuda", dtype=dtype))) for _ in range(3)]
                for down, up in pairs:
                    init_manifold_(down, up)
                groups = [{"params": [p for pair in pairs for p in pair], "lora_pair": True}]
                optimizer = RiemannionOT(groups, lr=1e-3)
                target = torch.randn(96, 80, device="cuda") * 0.01

                def train_step(optimizer, pairs=pairs, target=target):
                    optimizer.zero_grad(set_to_none=True)
                    loss = sum(((up.float() @ down.float() - target) ** 2).sum() for down, up in pairs)
                    loss.backward()
                    optimizer.step()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p).all() for pair in pairs for p in pair))
                    return loss.item()

                initial_loss = train_step(optimizer)  # Build the batched state before injecting a decomposition failure.
                with patch("torch.linalg.svd", side_effect=torch.linalg.LinAlgError("SVD did not converge")):
                    train_step(optimizer)
                for _ in range(6):
                    train_step(optimizer)
                saved = copy.deepcopy(optimizer.state_dict())
                resumed = RiemannionOT(groups, lr=1e-3)
                resumed.load_state_dict(saved)
                restored = resumed.state_dict()
                for key, state in saved["state"].items():
                    for name, value in state.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(restored["state"][key][name], value, rtol=0, atol=0)
                optimizer = resumed
                self.assertLess(train_step(optimizer), initial_loss)
                for state in optimizer.state_dict()["state"].values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
