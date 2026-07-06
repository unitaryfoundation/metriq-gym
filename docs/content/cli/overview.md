# CLI Overview

Metriq-Gym provides a command-line interface (`mgym`) for dispatching, monitoring, and uploading quantum benchmark results.

## Installation

The CLI is installed automatically with metriq-gym:

```bash
pip install metriq-gym
```

## Command Structure

```
mgym <resource> <action> [arguments] [options]
```

Resources:
- `job` - Individual benchmark jobs
- `suite` - Collections of benchmark jobs

## Quick Reference

| Command | Description |
|---------|-------------|
| `mgym job dispatch` | Dispatch a benchmark job to a quantum device or simulator. |
| `mgym job estimate` | Estimate circuit resource requirements before dispatching jobs. |
| `mgym job poll` | Poll job status and retrieve results when complete. |
| `mgym job view` | View job details and metadata. |
| `mgym job delete` | Delete a job from the local database. |
| `mgym job upload` | Upload job results to GitHub via pull request. |
| `mgym suite dispatch` | Dispatch a suite of benchmark jobs to a quantum device. |
| `mgym suite poll` | Poll suite jobs and retrieve results when complete. |
| `mgym suite view` | View jobs in a suite. |
| `mgym suite delete` | Delete all jobs in a suite from the local database. |
| `mgym suite upload` | Upload suite results to GitHub via pull request. |

## Getting Help

```bash
# Main help
mgym --help

# Job commands help
mgym job --help

# Specific command help
mgym job dispatch --help
```
