# Testing

Install the development tools in the project's Python environment, then check the
entire codebase and run the local suite from the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m ruff check modules scripts tests
$env:QT_QPA_PLATFORM = "offscreen"
$env:HF_HUB_OFFLINE = "1"
.\venv\Scripts\python.exe -m unittest discover -s tests
```

For a coverage report, install the development requirements and run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m coverage run --source=modules -m unittest discover -s tests
.\venv\Scripts\python.exe -m coverage report
```

The pull request smoke job runs Ruff, dependency-free tests, and Python syntax checks. Ruff's version is pinned in `requirements-dev.txt` and `.pre-commit-config.yaml`; update them together. The complete local suite also exercises Qt, PyTorch, and model-specific code through the installed environment, including both Anima VAEs, RGB/RGBA data handling, model exports, and embedding persistence. It uses small local fixtures and does not download model weights. Add a focused regression test whenever a bug is reproduced. Keep pure logic tests in the smoke job so they run on every pull request.

The full local suite covered 45% of executable lines in `modules` on 2026-09-24. This is a baseline, not a quality target: many model variants and GPU training paths are not exercised by the current tests. The next coverage priorities are dataset statistics and loading, CPU-testable model setup logic, and training orchestration. Measure behavior through real inputs and observable outputs; importing a module alone is not a meaningful coverage gain.

## Riemannion numerical regression checks

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_riemannion*py" -v
```

These checks exercise failed Cholesky factorizations with finite outputs, repeated
and extreme singular values, the float64 SciPy `gesvd` fallback, non-finite gradient
handling, and exact FP32 optimizer-state restoration for FP16/BF16 parameters.
When CUDA is available, a small rank-32 LoRA fixture also runs batched optimizer
steps and checkpoint resume in FP32 and BF16, including a forced SVD failure.
The GPU test is skipped on CPU-only hosts. It does not reproduce a full-size,
multi-hour training run.

Riemannion skips the entire optimizer step when gradients contain NaN or Inf,
leaving weights and optimizer state intact and issuing a warning. Repeated skips
require investigating the training loss, precision, and learning rate. A failure
on non-finite SVD input is reported explicitly; invalid matrices are never silently
replaced with zeros.
