# GitHub Integration

Metriq-Gym uploads benchmark results to GitHub via pull requests, enabling community contribution to the [metriq-data](https://github.com/unitaryfoundation/metriq-data) benchmark database.

## Overview

The upload workflow:
1. Creates a fork of the target repository (if needed)
2. Creates a new branch with your results
3. Opens a pull request for review
4. Once merged, results appear on [metriq.info](https://metriq.info)

## Setup

### 1. Create a GitHub Token

Create a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens).

**For Classic Tokens:**
- Select the `repo` scope

**For Fine-Grained Tokens:**
- Resource owner: Your account
- Repository access: All repositories (to cover future forks)
- Permissions:
  - Contents: Read and write
  - Pull requests: Read and write

### 2. Configure Environment

Add to your `.env` file:

```bash
GITHUB_TOKEN="your-token-here"

# Optional: customize upload destination
MGYM_UPLOAD_REPO="unitaryfoundation/metriq-data"
MGYM_UPLOAD_BASE_BRANCH="main"
MGYM_UPLOAD_DIR=""
```

## Upload Commands

### Single Job

```bash
mgym job upload <JOB_ID>
```

### Full Suite

```bash
mgym suite upload <SUITE_ID>
```

## Upload Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--repo` | `MGYM_UPLOAD_REPO` | `unitaryfoundation/metriq-data` | Target repository |
| `--dir` | `MGYM_UPLOAD_DIR` | `metriq-gym/v<version>/<provider>/<device>` | Directory path |

### Examples

```bash
# Default upload
mgym job upload abc123

# Custom repository
mgym job upload abc123 --repo myorg/my-data

# Custom directory
mgym job upload abc123 --dir results/2025/january
```

## Upload Process

### 1. Fork Creation

If you don't have a fork of the target repository, Metriq-Gym automatically creates one.

### 2. Branch Creation

A branch named `mgym/upload-<job_id>` is created with your results.

### 3. File Structure

Results are stored as JSON files:

```
metriq-gym/
└── v0.3/
    └── ibm/
        └── ibm_sherbrooke/
            └── 2025-01-15T12:00:00_BSEQ_abc123.json
```

### 4. Pull Request

A PR is opened with:
- Title: `mgym upload: <benchmark> on <provider>/<device>`
- Labels: `data`, `source:metriq-gym`

## Result File Format

```json
[
  {
    "app_version": "0.3.1",
    "timestamp": "2025-01-15T12:00:00.000000",
    "platform": {
      "provider": "ibm",
      "device": "ibm_sherbrooke"
    },
    "job_type": "BSEQ",
    "results": {
      "values": {
        "largest_connected_size": 100,
        "fraction_connected": 0.7874
      },
      "uncertainties": {
        "fraction_connected": 0.01
      }
    },
    "params": {
      "benchmark_name": "BSEQ",
      "shots": 1000
    }
  }
]
```

## Contributing to metriq-data

### First-Time Contributors

1. Your fork is created automatically
2. Review the PR before requesting merge
3. Maintainers review and merge contributions

### Repeat Contributors

Subsequent uploads use your existing fork and create new branches.

## Troubleshooting

### "GitHub token not provided"

Ensure `GITHUB_TOKEN` is set:

```bash
export GITHUB_TOKEN="your-token"
# or add to .env file
```

### Fork Creation Failed

If automatic forking fails:
1. Manually fork [unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data)
2. Retry the upload command

### Permission Denied

Verify your token has the required permissions:
- `repo` scope for classic tokens
- Contents + Pull requests for fine-grained tokens

### PR Creation Failed

If PR creation fails, the command returns a compare URL. Open it in your browser to manually create the PR.

## Best Practices

1. **Review before upload**: Verify results with `mgym job poll --json` first
2. **Don't upload test runs**: Only upload meaningful benchmark results
3. **Include metadata**: Results automatically include version, timestamp, and configuration
4. **One PR per upload**: Each upload creates a separate PR for easy review

## External Contributors

If you're contributing from outside the Unitary Foundation:

1. Fork `unitaryfoundation/metriq-data` manually (recommended)
2. Set `MGYM_UPLOAD_REPO` to your fork
3. Create PRs from your fork to the upstream repository

## Programmatic Upload

Use the Python API for custom upload workflows:

```python
from types import SimpleNamespace
from metriq_gym.run import upload_job
from metriq_gym.job_manager import JobManager

job_manager = JobManager()

upload_config = SimpleNamespace(
    job_id="your-job-id",
    repo="unitaryfoundation/metriq-data",
    dir=None,  # use default
)

upload_job(upload_config, job_manager)
```
