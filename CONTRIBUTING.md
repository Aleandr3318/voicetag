# Contributing to voicetag

Thank you for your interest in contributing to voicetag! This guide will help you get set up and explain the conventions we follow.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gr122lyBr/voicetag.git
cd voicetag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev,ml]"
```

This installs the package in editable mode along with all development tools (pytest, black, ruff, mypy, pre-commit) and the ML dependencies (pyannote.audio, resemblyzer, torch).

## Running Tests

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=voicetag --cov-report=term-missing

# Run a specific test file
pytest tests/test_models.py

# Run tests matching a keyword
pytest -k "test_enroll"
```

All tests are self-contained and use mocks -- no ML models, GPU, or audio hardware required.

## Code Style

We use the following tools to maintain consistent code quality:

### Black (formatter)

```bash
black voicetag/ tests/
```

Configuration: line length 99, target Python 3.9. See `[tool.black]` in `pyproject.toml`.

### Ruff (linter)

```bash
ruff check voicetag/ tests/
ruff check --fix voicetag/ tests/  # auto-fix where possible
```

Configuration: line length 99, select rules E/F/I/W. See `[tool.ruff]` in `pyproject.toml`.

### mypy (type checker)

```bash
mypy voicetag/
```

Configuration: Python 3.9, Pydantic plugin enabled, missing imports ignored. See `[tool.mypy]` in `pyproject.toml`.

## Pre-commit Hooks

We recommend setting up pre-commit hooks to catch issues before they reach CI:

```bash
pip install pre-commit
pre-commit install
```

Create a `.pre-commit-config.yaml` if one does not exist:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.0]
```

After installing, hooks will run automatically on every commit.

## Pull Request Guidelines

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```

2. **Write tests** for any new functionality. We aim to keep coverage above 80%.

3. **Follow existing code style.** Run `black`, `ruff`, and `mypy` before committing.

4. **Keep PRs focused.** One feature or fix per pull request. If you find an unrelated issue, open a separate PR.

5. **Write a clear PR description** explaining what the change does and why.

6. **Ensure all tests pass** before requesting review:
   ```bash
   pytest
   ruff check voicetag/ tests/
   black --check voicetag/ tests/
   mypy voicetag/
   ```

## Filing Issues

When filing a bug report, please include:

- Python version (`python --version`)
- voicetag version (`voicetag version` or `python -c "import voicetag; print(voicetag.__version__)"`)
- Operating system
- Minimal reproduction steps
- Full error traceback

For feature requests, describe the use case and expected behavior.

## Project Structure

```
voicetag/
  __init__.py       # Public API exports
  pipeline.py       # Core orchestration (VoiceTag class)
  encoder.py        # Resemblyzer wrapper & enrollment store
  diarizer.py       # Pyannote.audio wrapper
  overlap.py        # Overlap detection and merging
  models.py         # Pydantic v2 data models
  cli.py            # Typer CLI
  utils.py          # Audio I/O utilities
  exceptions.py     # Custom exception hierarchy

tests/
  conftest.py       # Shared fixtures
  test_models.py    # Model validation tests
  test_utils.py     # Audio utility tests
  test_encoder.py   # Encoder tests
  test_pipeline.py  # Pipeline integration tests
  test_cli.py       # CLI tests
  test_overlap.py   # Overlap detection tests
```

## Code of Conduct

Be kind, respectful, and constructive. We are all here to build something useful together.
