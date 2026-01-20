# CLI Overview

Metriq-Gym provides the `mgym` command-line interface for dispatching, monitoring, and managing benchmark jobs.

## Command Structure

```
mgym <resource> <action> [options]
```

Resources:

- `job` - Single benchmark operations
- `suite` - Benchmark suite operations

## Quick Reference

### Job Commands

| Command | Description |
|---------|-------------|
| `mgym job dispatch <config>` | Dispatch a benchmark job |
| `mgym job poll [job_id]` | Poll job status and retrieve results |
| `mgym job view [job_id]` | View job details |
| `mgym job upload <job_id>` | Upload results to GitHub |
| `mgym job delete [job_id]` | Delete a job from local database |
| `mgym job estimate <config>` | Estimate resource usage |

### Suite Commands

| Command | Description |
|---------|-------------|
| `mgym suite dispatch <suite.json>` | Dispatch all benchmarks in a suite |
| `mgym suite poll [suite_id]` | Poll suite status |
| `mgym suite view [suite_id]` | View all jobs in a suite |
| `mgym suite upload <suite_id>` | Upload all suite results |
| `mgym suite delete [suite_id]` | Delete all jobs in a suite |

## Global Options

| Option | Description |
|--------|-------------|
| `--provider`, `-p` | Provider name (ibm, ionq, braket, azure, quantinuum, origin, local) |
| `--device`, `-d` | Device identifier |
| `--help`, `-h` | Show help message |

## Interactive Mode

When job or suite IDs are omitted, commands enter interactive mode:

```bash
# Lists all jobs and prompts for selection
mgym job poll

# Lists all jobs for viewing
mgym job view
```

## Environment Variables

Configure default behavior via environment variables or `.env` file:

| Variable | Description |
|----------|-------------|
| `MGYM_LOCAL_DB_DIR` | Directory for job database |
| `MGYM_LOCAL_SIMULATOR_CACHE_DIR` | Cache directory for simulator results |
| `GITHUB_TOKEN` | Token for GitHub uploads |
| `MGYM_UPLOAD_REPO` | Default upload repository |

See [Provider Configuration](../providers/overview.md) for provider-specific variables.

## Local Job Database

The CLI maintains a local job database (`.metriq_gym_jobs.jsonl`) to track jobs between dispatch and poll operations.

!!! warning
    The local database is meant as a transient queue, not archival storage. Export results regularly or back up the file if you need to retain history.

Database location:

- Custom: Set `MGYM_LOCAL_DB_DIR`
- Default: Platform-specific user data directory (e.g., `~/Library/Application Support/metriq-gym` on macOS)

## Next Steps

- [Job Commands](job-commands.md) - Detailed job command reference
- [Suite Commands](suite-commands.md) - Working with benchmark suites
- [Resource Estimation](estimation.md) - Estimating job costs
