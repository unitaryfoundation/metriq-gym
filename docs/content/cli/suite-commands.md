# Suite Commands

Commands for estimating, dispatching, monitoring, and managing benchmark suites.

## dispatch

Dispatch a suite of benchmark jobs to a quantum device.

```bash
mgym suite dispatch <suite_config> [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_CONFIG` | STR | Path to suite configuration file or bundled suite name | Yes |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--provider, -p` | STR | Provider name (e.g., ibm, aws, azure, ionq, local; braket aliases aws) | `None` |
| `--device, -d` | STR | Device identifier | `None` |
| `--component, -c` | LIST | Component to dispatch; repeat the option to select multiple components | `None` |
| `--all` | BOOL | Explicitly dispatch every component, including guarded full suites | `False` |

### Bundled suites and JSON files

The suite argument accepts either a path to a JSON file or the name of a suite
bundled with metriq-gym. For example, `metriq_score_1_0` resolves to the
versioned Metriq Score 1.0 definition from any working directory.

### Dispatch selected components

Use `--component` (or `-c`) to dispatch one component and all of its configured
scale points:

```bash
mgym suite dispatch metriq_score_1_0 \
    --component qft --provider <provider> --device <device>
```

The option is repeatable:

```bash
mgym suite dispatch metriq_score_1_0 \
    -c bseq -c wit --provider <provider> --device <device>
```

Unknown or repeated component names are rejected before provider
initialization.

### Dispatch a complete guarded suite

Some suite definitions require an explicit `--all` opt-in before every
configured job is dispatched:

```bash
mgym suite dispatch metriq_score_1_0 \
    --all --provider <provider> --device <device>
```

Without `--component` or `--all`, a guarded suite prints its runtime and cost
warning, lists the available components, and exits before provider
initialization.

See [Metriq Score 1.0 Suite](../suites/metriq-score-1.0.md) for the versioned
component and scale-point reference.

---

## estimate

Estimate circuit resources and costs for a suite before dispatching jobs. The
command prints the total for the selection, without submitting jobs or recording
them in the local database. Add `-v` or `--verbose` for per-job estimates, circuit
resources and statistics, and estimation assumptions.

```bash
mgym suite estimate <suite_config> [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_CONFIG` | STR | Path to suite configuration file or bundled suite name | Yes |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--provider, -p` | STR | Provider name (e.g., ibm, aws, azure, ionq, local; braket aliases aws) | `None` |
| `--device, -d` | STR | Device identifier for device-dependent resource estimates | `None` |
| `--component, -c` | LIST | Component to estimate; repeat the option to select multiple components | `None` |
| `--all` | BOOL | Explicitly estimate every component, including guarded full suites | `False` |
| `--verbose, -v` | BOOL | Show per-job resources, circuit statistics, and estimation assumptions | `False` |

### Estimate the complete Metriq Score suite

```bash
mgym suite estimate metriq_score_1_0 \
    --provider ibm --device ibm_phoenix --all
```

### Estimate selected components

```bash
mgym suite estimate metriq_score_1_0 \
    --provider ibm --device ibm_phoenix -c qft -c wit
```

Component selection follows the same rules as `suite dispatch`: a component
includes all of its configured scale points, unknown or repeated names are
rejected, and guarded suites require `--component` or `--all`.

The output starts with a short header identifying the suite, provider, and device.
The command uses the same estimator as `job estimate`:

- `--provider ibm` shows total estimated QPU time in seconds. With `--verbose`,
  each job's estimate is also shown in seconds. Circuits are transpiled and
  scheduled locally using timing metadata from the selected device. Provider
  access is needed to read the metadata, but
  nothing is submitted to or executed on the QPU.
- `--provider quantinuum` shows total H-series Quantum Credits (HQCs).
- `--provider aws` (or its alias `braket`) shows total shots and total tasks,
  counting one task per circuit. Use `--verbose` for the full resource breakdown.
- Other providers report that no cost estimate is available. Use `--verbose` to
  show circuit resources.

IBM estimates assume the default repetition delay, initial reset, and about two
seconds of overhead per circuit batch, treating each batch as one execution
sub-job. Additional server-side splitting and runtime mitigation are excluded.
The estimate describes QPU usage, not queue time, wall-clock completion time,
or a monetary price. Sessions can also bill idle time between executions, which
cannot be determined statically and is excluded. See
[IBM's usage estimation guide](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time).

Some benchmarks need a device to determine resources such as available qubit
pairs. If a job cannot be estimated, the command reports the failure and
continues with the remaining jobs. Totals are marked as partial when any job
fails and cover only successfully estimated jobs.

If IBM timing estimation fails, QPU usage is marked unavailable when no timings
are available, or partial when only some selected jobs have timings. Use
`--verbose` for the reasons and any available circuit resource counts. Omitting
`--device` also leaves IBM timing unavailable.

---

## poll

Poll suite jobs and retrieve results when complete.

```bash
mgym suite poll [suite_id] [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_ID` | STR | Suite ID to poll | No |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--json` | STR | Export results to JSON file | `None` |
| `--no-cache` | BOOL | Ignore locally cached results and refetch | `False` |

### Export results to JSON

Write the suite's completed results to a file:

```bash
mgym suite poll <suite_id> --json suite-results.json
```

The file contains a JSON array in suite job order. Each record uses the same
format as a completed job in `mgym suite upload`, including metadata, parameters,
results, and a UTC timestamp. Without `--json`, results are displayed as a table.

Failed jobs are reported and skipped. If any job is still pending, or no
completed jobs remain, no file is written.

---

## view

View jobs in a suite.

```bash
mgym suite view [suite_id]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_ID` | STR | Suite ID to view | No |

---

## delete

Delete all jobs in a suite from the local database.

```bash
mgym suite delete [suite_id]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_ID` | STR | Suite ID to delete | No |

---

## upload

Upload suite results to GitHub via pull request.

```bash
mgym suite upload [suite_id] [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_ID` | STR | Suite ID to upload | No |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--repo` | STR | Target GitHub repo (owner/repo) (env: `MGYM_UPLOAD_REPO`) | `unitaryfoundation/metriq-data` |
| `--base` | STR | Base branch for the PR (env: `MGYM_UPLOAD_BASE_BRANCH`) | `main` |
| `--dir` | STR | Directory in repo for the JSON file (env: `MGYM_UPLOAD_DIR`) | `None` |
| `--branch` | STR | Branch name for the PR | `None` |
| `--title` | STR | Pull request title | `None` |
| `--body` | STR | Pull request body | `None` |
| `--commit-message` | STR | Commit message | `None` |
| `--clone-dir` | STR | Working directory to clone into (env: `MGYM_UPLOAD_CLONE_DIR`) | `None` |
| `--dry-run` | BOOL | Do not push or open a PR; print actions only | `False` |

Jobs in the suite that failed (dispatch raised, or the provider reported a failure)
are included as `outcome: "error"` records with the captured error, so one failed
benchmark does not block the upload. Jobs that are still pending do.

---
