# The Metriq Ecosystem

Metriq-Gym is part of a larger ecosystem for tracking quantum computing progress. Understanding this ecosystem helps you see how your benchmark results contribute to the community.

## Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ metriq-gym  │ ──▶ │ metriq-data │ ──▶ │ metriq.info │
│  (collect)  │     │   (store)   │     │  (display)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

The ecosystem consists of three interconnected components:

1. **Metriq-Gym** (this project): Run standardized benchmarks on quantum hardware and simulators
2. **[metriq-data](https://github.com/unitaryfoundation/metriq-data)**: GitHub repository storing all community benchmark results
3. **[metriq.info](https://metriq.info)**: Web platform for exploring and comparing results

## How It Works

### 1. Run Benchmarks

You run benchmarks locally using Metriq-Gym:

```bash
mgym job dispatch config.json --provider ibm --device ibm_fez
mgym job poll latest
```

### 2. Upload Results

Upload your results to the community database:

```bash
mgym job upload <JOB_ID>
```

This creates a pull request to `metriq-data`. Once reviewed and merged, your results become part of the public dataset.

### 3. Explore on metriq.info

Visit [metriq.info](https://metriq.info) to:

- Compare benchmark results across devices
- Track performance trends over time
- Explore community contributions
- Download datasets for analysis

!!! note
    The Metriq platform is currently available at [beta.metriq.info](https://beta.metriq.info). The URL will transition to [metriq.info](https://metriq.info) at launch.

## Why Contribute?

Community benchmark data helps:

- **Researchers**: Track quantum hardware progress over time
- **Developers**: Choose the right hardware for their applications  
- **Hardware providers**: Understand competitive positioning
- **Everyone**: Build a transparent record of quantum computing capabilities

## Data Quality Guidelines

To ensure your contributions are valuable:

**Do:**

- Run benchmarks with sufficient shots for statistical significance
- Use recommended trial counts
- Verify results before uploading
- Include all relevant configuration parameters

**Don't:**

- Upload test runs or debugging data
- Cherry-pick favorable results
- Modify result files manually
- Upload duplicates

## Related Links

- [metriq-data Repository](https://github.com/unitaryfoundation/metriq-data) - Browse raw benchmark data
- [Metriq Platform](https://metriq.info) - Explore results visually
- [Unitary Foundation](https://unitary.foundation) - Organization behind Metriq
- [GitHub Upload Guide](uploading/github.md) - Detailed upload instructions
