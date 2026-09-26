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
