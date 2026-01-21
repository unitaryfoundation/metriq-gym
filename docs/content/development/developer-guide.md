# Developer Guide

This guide covers setting up a development environment for contributing to Metriq-Gym.

## Prerequisites

Before you begin, ensure you have:

- [Python](https://www.python.org/downloads/) (version 3.12 or newer)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- [Git](https://git-scm.com/) for version control

## Getting Started

### Clone the Repository

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/unitaryfoundation/metriq-gym.git
cd metriq-gym
```

If you already have a clone, update it:

```bash
git pull --recurse-submodules
```

### Install Dependencies

Install all dependencies including dev tools:

```bash
uv sync --all-groups
```

This creates a virtual environment in `.venv` and installs all dependencies.

### Activate Environment

Either activate the virtual environment:

```bash
source .venv/bin/activate
```

Or use `uv run` for isolated commands:

```bash
uv run pytest
```

## macOS Note

macOS users installing optional `pyqpanda3` support must install `libidn2`:

```bash
brew reinstall libidn2
```

Install this **before** running `uv sync` to avoid build errors.

## Development Workflow

### Pre-commit Hooks

Install pre-commit hooks after cloning:

```bash
uv run pre-commit install
```

This runs linting and formatting automatically on each commit.

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m "not e2e"

# End-to-end tests only
uv run pytest -m e2e

# Specific test file
uv run pytest tests/test_benchmarks.py
```

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Check types with mypy
uv run mypy
```

### Building Documentation

```bash
cd docs
uv run mkdocs serve
```

Open `http://127.0.0.1:8000` to view the documentation locally.

## Project Structure

```
metriq-gym/
├── metriq_gym/
│   ├── benchmarks/      # Benchmark implementations
│   ├── exporters/       # Result export (JSON, GitHub PR)
│   ├── local/           # Local simulator provider
│   ├── origin/          # OriginQ provider
│   ├── quantinuum/      # Quantinuum provider
│   ├── schemas/         # JSON schemas and examples
│   ├── cli.py           # CLI argument parsing
│   ├── constants.py     # JobType enum, schema mapping
│   ├── job_manager.py   # Job tracking
│   ├── registry.py      # Benchmark registration
│   └── run.py           # Main entrypoint
├── tests/               # Test suite
├── docs/                # Documentation
├── submodules/          # External dependencies
└── pyproject.toml       # Project configuration
```

## Contributing

### Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add support for new benchmark type
fix: correct quantum volume calculation
docs: update provider setup guide
refactor: simplify job manager interface
```

### Pull Request Guidelines

- Rebase onto the latest `main` before opening a PR
- Link related issues or discussions
- Include CLI output or screenshots for user-facing changes
- Ensure all tests pass
- Get approval from maintainers

## Code Style

The project uses:
- **Ruff** for linting and formatting
- **mypy** for type checking
- Line length: 100 characters

### Type Annotations

Use type annotations for all public functions:

```python
def dispatch_job(config: SimpleNamespace, job_manager: JobManager) -> str:
    """Dispatch a benchmark job.

    Args:
        config: Job configuration with provider, device, and config path
        job_manager: Job tracking instance

    Returns:
        The metriq-gym job ID
    """
    ...
```

## Testing

### Test Categories

| Marker | Description |
|--------|-------------|
| (none) | Unit tests (fast, no external dependencies) |
| `e2e` | End-to-end tests (may require credentials) |

### Writing Tests

```python
import pytest
from metriq_gym.benchmarks.wit import WIT

def test_wit_dispatch():
    """Test WIT benchmark dispatch creates correct circuits."""
    benchmark = WIT(config)
    result = benchmark.dispatch_handler(mock_device)
    assert result.circuits is not None

@pytest.mark.e2e
def test_wit_full_workflow():
    """Test full WIT workflow on simulator."""
    # This test requires local simulator
    ...
```

## Debugging

### Verbose Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Local Testing

Test against the local simulator:

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    -p local -d aer_simulator
```

## Release Process

Releases are managed by maintainers:

1. Version is determined by `setuptools_scm` from git tags
2. CI builds and publishes to PyPI
3. Documentation is deployed to GitHub Pages

## Getting Help

- [Issue Tracker](https://github.com/unitaryfoundation/metriq-gym/issues)
- [Discussions](https://github.com/unitaryfoundation/metriq-gym/discussions)
- [CONTRIBUTING.md](https://github.com/unitaryfoundation/metriq-gym/blob/main/CONTRIBUTING.md)
