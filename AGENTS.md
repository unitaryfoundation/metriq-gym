# AGENTS.md

Read and follow [AI_POLICY.md](AI_POLICY.md) and [CONTRIBUTING.md](CONTRIBUTING.md)
before making changes or preparing a contribution. metriq-gym implements quantum
benchmarks; preserving their scientific meaning matters as much as working code.

## Development workflow

- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup. Read Python requirements,
  dependencies, and tool settings from [pyproject.toml](pyproject.toml) and resolved
  dependency versions from [uv.lock](uv.lock).
- Use [.github/workflows/](.github/workflows/) and
  [.pre-commit-config.yaml](.pre-commit-config.yaml) for current build and check
  commands.
- Initialize submodules at their recorded revisions rather than updating them
  during setup. Report missing prerequisites instead of silently changing the host.

## Tests and checks

- Start with tests covering the changed behavior.
- For CLI, dispatch/poll, or schema/example changes, run the relevant local
  end-to-end tests. Use the full suite for broad changes or when requested.
- Run the applicable lint, formatting, type, and dependency checks defined in the
  configurations above. Format only touched files.
- Build distributions for packaging changes and the documentation site for changes
  under `docs/`. For root Markdown or template changes, check content, local links,
  and `git diff --check`; runtime tests are unnecessary unless behavior changes too.
- Report checks and outcomes accurately, including failures, skipped checks, and
  limitations. In PR descriptions, follow the reporting guidance below. Do not
  claim validation that was not performed.

## Benchmark verification

- Cover behavior changes with appropriate regression or feature tests. For a bug
  fix, establish that the reproducer or regression test fails on the old behavior
  and passes with the fix.
- Justify numerical expectations independently: use an analytic result, an ideal
  small-instance calculation, or a trusted reference implementation. Check the
  benchmark protocol and assumptions against the cited source.
- Use fixed seeds where supported and explicit, justified statistical tolerances.
  Record parameters, shots, and relevant software and backend settings.
- Do not weaken assertions, increase tolerances, skip failures, or suppress lint
  findings merely to make checks pass. Explain any justified changes to tests,
  benchmark parameters, scoring, or measurement interpretation and their effect
  on comparisons with earlier results.
- The local end-to-end tests use reduced instances and simulated backends. Treat
  them as smoke checks, not evidence of full-scale hardware performance. Use
  offline fixtures or local simulators for routine verification.

## Agent boundaries

- Follow [AI_POLICY.md](AI_POLICY.md#boundaries-for-agents) for approval required
  before opening agent-driven PRs that change dependencies, submodule revisions,
  CI, releases, or verification workflows, or involve protected resources.
- Obtain authorization before credentialed, paid-service, or hardware operations.
  Do not expose credentials or use a real backend for verification without that
  authorization. Do not access private credential information or implicitly
  include it in any session context.
- Do not commit, push, open a PR, or post comments unless explicitly requested.
  Commands such as `mgym job upload` and `mgym suite upload` can publish data PRs;
  they are not local verification steps.
- Keep changes scoped to the task. Safe local investigation and drafting may
  proceed while approval for a later action is pending. Existing approval for
  the same scope remains valid; explain any action that still needs approval.

## PR descriptions

- Start with a short explanation of what changed and why. Keep relevant risks,
  limitations, and effects on benchmark scores or comparability visible.
- Put lengthy supporting evidence in an optional `<details>` block with a
  descriptive `<summary>`, such as "Reproduction details" or "Numerical validation".
  Short PRs do not need one. Human review must cover collapsed content too.
- Include an accurate, visible `AI use:` note naming each AI tool and model used,
  and their role, including PR preparation or creation. Use identifiers reported
  by the tool or session; if the model is not exposed, say so rather than guessing.
  Leave statements about human review and approval to the responsible human.
- Do not include passing-test counts or metadata already available in GitHub CI
  reports, including in collapsed sections; link to the reports when useful.
  Describe only validation evidence that CI does not provide, such as manual checks
  or scientific reasoning.

## Repository conventions

See [CONTRIBUTING.md](CONTRIBUTING.md#repository-layout) for the repository layout
and coding standards.

- `metriq_gym/suites/metriq_score_1_0.json` defines the canonical suite. Example
  configurations and reduced test fixtures are not substitutes for that definition.
- Preserve runtime device identifiers and the meaning of exported results. Review
  downstream compatibility when changing schemas, metadata, or score semantics.
- Do not hand-edit generated `metriq_gym/_version.py`. If asked to commit, follow
  the Conventional Commit prefixes described in `CONTRIBUTING.md`.

Adapted from the [AI4QOSS AGENTS.md template](https://github.com/unitaryfoundation/ai4qoss/blob/main/standards/agents-md-template.md).
