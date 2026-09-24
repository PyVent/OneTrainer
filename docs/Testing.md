# Testing

Run the complete local suite with the project's Python environment:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

For a coverage report, install the development requirements and run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m coverage run --source=modules -m unittest discover -s tests
.\venv\Scripts\python.exe -m coverage report
```

The pull request smoke job uses dependency-free tests and Python syntax checks. The complete local suite also exercises Qt, PyTorch, and model-specific code through the installed environment. Add a focused regression test whenever a bug is reproduced. Keep pure logic tests in the smoke job so they run on every pull request.

The full local suite covered 45% of executable lines in `modules` on 2026-09-24. This is a baseline, not a quality target: many model variants and GPU training paths are not exercised by the current tests. The next coverage priorities are dataset statistics and loading, CPU-testable model setup logic, and training orchestration. Measure behavior through real inputs and observable outputs; importing a module alone is not a meaningful coverage gain.
