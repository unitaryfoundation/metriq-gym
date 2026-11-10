# Contributing to metriq-gym

Thanks for your interest in improving the project! This guide describes how to get set up, make changes, and submit them for review. For deeper technical context, consult the [full documentation](https://unitaryfoundation.github.io/metriq-gym/).

Read the [AI contribution policy](AI_POLICY.md) before contributing. It covers
scientific correctness, reproducible verification, AI disclosure, human review,
and authorization for agent-driven work. Its evidence requirements apply to all
contributions, however they were produced.

Coding-agent instructions start in [AGENTS.md](AGENTS.md). If your tool does not
load that file automatically, explicitly provide it and the linked policy.

## Getting started

1. Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository with submodules:
   ```sh
   git clone --recurse-submodules https://github.com/unitaryfoundation/metriq-gym.git
   cd metriq-gym
   ```
3. Install dependencies and activate the virtual environment:
   ```sh
   uv sync --all-groups
   source .venv/bin/activate  # or use `uv run` for one-off commands
   ```
4. Install the pre-commit hooks so linting and type checks run automatically:
   ```sh
   uv run pre-commit install
   ```

## Repository layout

The main directories are:

```text
metriq-gym/
|-- .github/workflows/       # CI, documentation, and release workflows
|-- docs/content/            # Documentation source
|-- metriq_gym/              # Python package and CLI
|   |-- benchmarks/          # Benchmark implementations and analysis
|   |-- dashboard/           # Local job dashboard
|   |-- exporters/           # Result output and data PR export
|   |-- local/               # Local simulator support
|   |-- qplatform/           # Device and job adapters
|   |-- schemas/             # Benchmark configuration schemas
|   |   `-- examples/        # Example configurations and circuit fixtures
|   `-- suites/              # Benchmark suite definitions
|-- submodules/              # External benchmark implementations
`-- tests/
    |-- unit/                # Unit tests
    `-- e2e/                 # Local CLI workflow tests
```

## Development workflow

- Use `uv run mgym --help` to confirm the CLI is wired correctly after changes.
- Keep provider credentials in a local `.env` copied from `.env.example`; never commit secrets.
- When touching schemas or benchmark examples, validate them with `uv run mgym job dispatch ...` using a local simulator before opening a PR.

## Coding standards

- Follow the defaults enforced by Ruff: 4-space indentation and 100-character lines.
- Run `uv run ruff check --fix` and `uv run ruff format` before committing.
- Maintain or extend type hints and verify them with `uv run mypy`.
- Add concise, purposeful docstrings; avoid large dumps of commented-out code.
- Keep provider-specific logic in backend adapters rather than adding provider
  branches throughout benchmark implementations.

## Testing checklist

Run the relevant test suites prior to submission:

```sh
uv run pytest tests/unit        # fast coverage
uv run pytest -m e2e            # CLI workflows (run before merging)
uv run pytest                   # full suite when feasible
```

For features that add new benchmark payloads or alter CLI flows, update example
configurations in `metriq_gym/schemas/examples/` and relevant test fixtures in
`tests/e2e/`. Document CLI changes under `docs/content/cli/` as appropriate.

## Commit and PR guidelines

- Use Conventional Commit prefixes (e.g. `fix:`, `feat:`, `chore:`) and keep each commit focused.
- Rebase on the latest `main` before raising a pull request.
- Keep PR descriptions concise: explain what changed and why, link relevant issues,
  and include CLI output or screenshots only when they help review the change.
- Do not include passing-test counts or repeat metadata already in GitHub CI
  reports. Link to those reports when useful; describe additional validation and
  explain failures, skipped checks, or limitations that affect review.
- Coordinate dependency bumps or submodule updates with maintainers. Agent-driven
  changes must follow the explicit approval requirements in [AI_POLICY.md](AI_POLICY.md#boundaries-for-agents).

## Questions and support

- Open a [discussion thread](https://github.com/unitaryfoundation/metriq-gym/discussions) for design questions or roadmap ideas.
- For quick help, join the `#metriq` channel on [Discord](http://discord.unitary.foundation).
- Security disclosures should be sent privately to the maintainers listed in `pyproject.toml`.
