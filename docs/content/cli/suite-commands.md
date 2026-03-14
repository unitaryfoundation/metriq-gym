# Suite Commands

Commands for dispatching, monitoring, and managing benchmark suites.

## dispatch

Dispatch a suite of benchmark jobs to a quantum device.

```bash
mgym suite dispatch <suite_config> [OPTIONS]
```

### Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `SUITE_CONFIG` | STR | Path to suite configuration file | Yes |

### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--provider, -p` | STR | Provider name (e.g., ibm, braket, azure, ionq, local) | `None` |
| `--device, -d` | STR | Device identifier | `None` |

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

---
