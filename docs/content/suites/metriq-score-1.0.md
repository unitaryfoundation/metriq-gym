# Metriq Score 1.0 Suite

Metriq Score 1.0 is a versioned definition of the canonical Metriq benchmark
suite. Later iterations can introduce new versioned definitions without
changing the meaning of this one.

The machine-readable definition is
[`metriq_gym/suites/metriq_score_1_0.json`](https://github.com/unitaryfoundation/metriq-gym/blob/main/metriq_gym/suites/metriq_score_1_0.json).

The definition contains eight score components and 22 benchmark jobs:

| Component | Jobs | Scale datapoints |
|-----------|-----:|------------------|
| BSEQ | 1 | All device qubit pairs |
| EPLG | 1 | 10, 20, 50, and 100-qubit subchains |
| Mirror Circuits | 6 | Widths 8, 16, 24, 32, 64, and 128 |
| CLOPS | 1 | 100 qubits |
| QML Kernel | 4 | 10, 20, 30, and 50 qubits |
| WIT | 1 | 7 qubits |
| LR-QAOA | 4 | 10, 20, 50, and 100 qubits |
| QFT | 4 | 4, 8, 12, and 20 qubits |

## Estimate resources before dispatch

Request estimates for all 22 jobs with:

```bash
mgym suite estimate metriq_score_1_0 --provider ibm --device ibm_phoenix --all
```

To estimate one component and all its scale points, replace `--all` with
`--component qft` (or another component name). Repeat `--component` to select
multiple components. A short header identifies the suite, provider, and device.
By default, IBM estimates show only the total QPU time in
seconds, using backend timing metadata and local compilation without submitting
any jobs. Quantinuum estimates show only total HQCs, while AWS estimates
(`--provider aws` or `--provider braket`) show total shots and total tasks,
counting one task per circuit. Add `-v` or `--verbose` for per-job estimates,
circuit resources and statistics, and estimation assumptions.

The suite's twirled CLOPS estimate uses the circuit template and configured
number of randomizations and shots. It excludes runtime twirling overhead and
idle time billed within a session, so actual IBM usage can be higher.

See [Suite Commands](../cli/suite-commands.md#estimate) for options and how
partial estimates are reported.

## References

- Cosentino et al., [“Metriq: A Collaborative Platform for Benchmarking
  Quantum Computers”](https://arxiv.org/abs/2603.08680v1), especially
  Section III (Metriq Score), Section IV (benchmark suite), and Appendix A
  (benchmark configuration reference).
