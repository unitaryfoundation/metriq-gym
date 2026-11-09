# AI contribution policy

AI tools are welcome in metriq-gym. Their use does not lower the standard for a
contribution: a human must understand the work, verify it, and take responsibility
for it. This policy covers code, tests, documentation, issues, pull requests,
reviews, and discussion comments.

Benchmark correctness is essential. An implementation can run successfully and
produce plausible numbers while measuring the wrong quantity. Errors in circuit
construction, measurement interpretation, or scoring can invalidate comparisons
between quantum devices. Passing CI alone does not establish scientific correctness.

## What a contribution must explain

These requirements apply whether or not AI was used, as applicable to the contribution:

- State the intended outcome and why the change is needed. An issue link alone
  does not explain what the change should achieve.
- For a bug report or fix, provide a minimal reproducer, the expected behavior,
  and the observed behavior. A fix should include a regression check where practical.
- For a new feature or method, state the functional or performance requirement
  and show how it was verified.
- Run the applicable checks in [CONTRIBUTING.md](CONTRIBUTING.md) before requesting
  review. For validation not covered by CI, supply reproducible evidence: commands,
  relevant configuration and environment, and the results actually obtained.
  Explain checks that were not run, failed, or require maintainer help; do not
  describe them as passing.

Keep PR descriptions concise: explain what changed, why, and anything reviewers
need to assess correctness or compatibility. Do not include passing-test counts
or repeat statuses, logs, or other metadata already available in GitHub CI reports.
Link to those reports when useful. Include only additional validation evidence
and limitations relevant to the change.

Documentation-only changes can be verified by checking their content and links;
they do not need new runtime tests. Maintainers may return contributions that lack
the explanation or evidence needed for review.

## Scientific validation

For changes to benchmark circuits, analysis, scoring, or execution semantics:

- Identify the benchmark protocol or reference and the assumptions the
  implementation relies on. Check citations against the original source.
- Justify expected results independently of the code being changed, using an
  analytic result, an ideal small-instance calculation, or a trusted reference
  implementation as appropriate. Tests generated alongside an implementation
  can repeat its mistakes.
- Record enough detail to reproduce numerical evidence: benchmark parameters,
  seeds where applicable, shots, software versions, and relevant device,
  compilation, and execution settings. For stochastic checks, explain tolerances
  and account for sampling uncertainty.
- Clearly distinguish synthetic fixtures, ideal or noisy simulations, and
  measurements from hardware. Reduced-size smoke tests are useful but do not
  validate full-scale benchmark performance. Hardware runs are not required for
  every contribution.
- Explain changes to the meaning of results or their comparability with earlier
  runs, including effects on the suite definition, schemas, and downstream data.
  Do not weaken assertions, relax tolerances, change benchmark parameters or score
  normalization, or discard failed runs merely to make a check pass or improve a
  reported score. Scientifically justified changes must be explicit and supported
  by evidence.

Never fabricate measurements, test output, execution claims, or references. If the
evidence is incomplete, state the limitation.

## AI disclosure and human review

In each pull request, state whether AI tools were used and what they did, such as
drafting code, generating tests, reviewing a diff, or running commands. Name the
tool and preserve any version or attribution it records; do not guess versions.
A short note is sufficient, for example:

> AI use: Codex drafted the implementation and tests; I reviewed the diff and
> verified the results described above.

Use that wording only if it accurately describes the human review and validation.
For issues, reviews, and discussions, disclose substantive AI assistance too.
When including generated analysis, label it clearly and add your own assessment
of its relevance and correctness. Do not answer reviewers by pasting raw or
lightly reviewed model output.

Before submitting, read the complete diff, simplify unnecessary complexity, and
be prepared to explain the implementation and its scientific reasoning. The
submitting human remains responsible for addressing review feedback. An agent's
self-review is not a substitute for human review.

We do not accept autonomous AI submissions without meaningful human involvement.
Maintainer-approved project automation may continue within its agreed scope.
AI assistance itself is not a reason to reject a contribution; misleading
disclosure or unsupported claims may be.

## Boundaries for agents

Obtain explicit maintainer approval before opening an agent-driven pull request
that adds or updates dependencies or submodule revisions, changes CI or release
configuration, or changes the workflows used to verify contributions.

Obtain explicit authorization before running operations that use credentials,
paid services, or hardware backends, and before opening a pull request containing
agent-driven changes that use those resources. Agree on the target and scope,
including any spending or job limits. A request to fix code is not permission to
spend hardware credits. Prefer local simulators and offline fixtures for routine
validation; never expose credentials in code, logs, or reports.

Existing approval for the same scope remains valid. Safe local investigation and
drafting can proceed while any required approval for later actions is pending.

## Policy sources

Adapted from the [AI4QOSS contribution policy](https://github.com/unitaryfoundation/ai4qoss/blob/main/standards/ai-contribution-policy.md),
with inspiration from [PyTorch's AI policy](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md).
