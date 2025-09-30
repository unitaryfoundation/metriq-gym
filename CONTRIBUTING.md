# Contributing to metriq-gym

Thanks for your interest in improving the project! This guide describes how to get set up, make changes, and submit them for review. For deeper technical context, consult the docs hosted at [metriq-gym.readthedocs.io](https://metriq-gym.readthedocs.io/).

## Getting started

1. Install Python 3.12 or newer and [Poetry](https://python-poetry.org/docs/#installation).
2. Clone the repository with submodules:
   ```sh
   git clone --recurse-submodules https://github.com/unitaryfoundation/metriq-gym.git
   cd metriq-gym
   ```
3. Install dependencies and activate the virtual environment:
   ```sh
   poetry install
   poetry shell  # or eval "$(poetry env activate)"
   ```
4. Install the pre-commit hooks so linting and type checks run automatically:
   ```sh
   poetry run pre-commit install
   ```

## Development workflow

- Use `poetry run mgym --help` to confirm the CLI is wired correctly after changes.
- Keep provider credentials in a local `.env` copied from `.env.example`; never commit secrets.
- When touching schemas or benchmark examples, validate them with `poetry run mgym job dispatch ...` using a local simulator before opening a PR.

## Coding standards

- Follow the defaults enforced by Ruff: 4-space indentation and 100-character lines.
- Run `poetry run ruff check --fix` and `poetry run ruff format` before committing.
- Maintain or extend type hints and verify them with `poetry run mypy`.
- Add concise, purposeful docstrings; avoid large dumps of commented-out code.
- Keep provider-specific logic inside the relevant subpackage (`local/`, `quantinuum/`, `qplatform/`).

## Testing checklist

Run the relevant test suites prior to submission:

```sh
poetry run pytest tests/unit        # fast coverage
poetry run pytest -m e2e            # CLI workflows (run before merging)
poetry run pytest                   # full suite when feasible
```

For features that add new benchmark payloads or alter CLI flows, update fixtures in `jobs/` and document the change in the `CLI Workflows` guide under `docs/source/` if appropriate.

## Commit and PR guidelines

- Use Conventional Commit prefixes (e.g. `fix:`, `feat:`, `chore:`) and keep each commit focused.
- Rebase on the latest `main` before raising a pull request.
- Include a clear description, linked issues, and CLI output or screenshots for user-facing changes.
- Confirm lint, type, and test commands in the PR template or description.
- Coordinate dependency bumps or submodule updates with maintainers when in doubt.

## Questions and support

- Open a [discussion thread](https://github.com/unitaryfoundation/metriq-gym/discussions) for design questions or roadmap ideas.
- For quick help, join the `#metriq` channel on [Discord](http://discord.unitary.foundation).
- Security disclosures should be sent privately to the maintainers listed in `pyproject.toml`.
