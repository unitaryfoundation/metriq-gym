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
| `mgym job dispatch` | Dispatch a benchmark job |
| `mgym job poll` | Poll job status and results |
| `mgym job view` | View job details |
| `mgym job delete` | Delete a job |
| `mgym job upload` | Upload results to GitHub |
| `mgym job estimate` | Estimate resource requirements |
| `mgym suite dispatch` | Dispatch a suite of jobs |
| `mgym suite poll` | Poll suite status |
| `mgym suite view` | View suite jobs |
| `mgym suite delete` | Delete a suite |
| `mgym suite upload` | Upload suite results |

## Getting Help

```bash
# Main help
mgym --help

# Job commands help
mgym job --help

# Specific command help
mgym job dispatch --help
```
