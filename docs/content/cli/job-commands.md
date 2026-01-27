# Job Commands

Commands for dispatching, monitoring, and managing individual benchmark jobs.

## dispatch

Dispatch a benchmark job to a quantum device or simulator.

```bash
mgym job dispatch <config> [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `CONFIG` | STR | Path to job configuration JSON file | Yes |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--provider, -p` | STR | Provider name (e.g., ibm, braket, azure, ionq, local) | `None` |
| `--device, -d` | STR | Device identifier | `None` |

---

## estimate

Estimate circuit resource requirements before dispatching jobs.

```bash
mgym job estimate <config> [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `CONFIG` | STR | Path to job configuration JSON file | Yes |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--provider, -p` | STR | Provider name (e.g., ibm, braket, azure, ionq, local) | `None` |
| `--device, -d` | STR | Device identifier | `None` |

---

## poll

Poll job status and retrieve results when complete.

```bash
mgym job poll [job_id] [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `JOB_ID` | STR | Job ID to poll (use 'latest' for most recent) | No |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--json` | STR | Export results to JSON file | `None` |
| `--no-cache` | BOOL | Ignore locally cached results and refetch | `False` |
| `--include-raw` | BOOL | Export raw measurement counts to a separate debug file | `False` |

### Debug Output

When `--include-raw` is used with `--json`, a separate debug file is created alongside the results file. For example:

```bash
mgym job poll latest --json result.json --include-raw
```

This creates:

- `result.json` - Standard benchmark results
- `result_debug.json` - Debug data for replay/debugging:

```json
{
    "job_id": "...",
    "job_type": "...",
    "params": {...},
    "job_data": {...},
    "raw_counts": [{"measurement_counts": {"00": 512, "11": 488}}, ...]
}
```

This is useful for debugging benchmark results locally without access to the original quantum provider.

!!! note
    If results are cached, raw counts are not available. Use `--no-cache` to refetch from the provider.

---

## view

View job details and metadata.

```bash
mgym job view [job_id]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `JOB_ID` | STR | Job ID to view (lists all if omitted) | No |

---

## delete

Delete a job from the local database.

```bash
mgym job delete [job_id]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `JOB_ID` | STR | Job ID to delete | No |

---

## upload

Upload job results to GitHub via pull request.

```bash
mgym job upload [job_id] [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `JOB_ID` | STR | Job ID to upload | No |

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

---
