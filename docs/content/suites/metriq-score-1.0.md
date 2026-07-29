# Metriq Score 1.0 Suite

The canonical configuration used for Metriq Score 1.0 is
[`metriq_gym/suites/metriq_score_1_0.json`](https://github.com/unitaryfoundation/metriq-gym/blob/main/metriq_gym/suites/metriq_score_1_0.json).
It formalizes the benchmark parameters and scale datapoints reported in the
[Metriq paper (v1)](https://arxiv.org/abs/2603.08680v1).

The JSON ships with metriq-gym, so `metriq_score_1_0` can be used as a bundled
suite name from any working directory. A direct path to a suite JSON file is
also accepted.

The suite contains eight score components and 22 dispatchable jobs:

| Component | `--component` value | Jobs | Scale datapoints |
|-----------|---------------------|-----:|------------------|
| BSEQ | `bseq` | 1 | All device qubit pairs |
| EPLG | `eplg` | 1 | 10, 20, 50, and 100-qubit subchains |
| Mirror Circuits | `mirror-circuits` | 6 | Widths 8, 16, 24, 32, 64, and 128 |
| CLOPS | `clops` | 1 | 100 qubits |
| QML Kernel | `qml-kernel` | 4 | 10, 20, 30, and 50 qubits |
| WIT | `wit` | 1 | 7 qubits |
| LR-QAOA | `lr-qaoa` | 4 | 10, 20, 50, and 100 qubits |
| QFT | `qft` | 4 | 4, 8, 12, and 20 qubits |

## Run a component

Pass `--component` (or `-c`) to dispatch one benchmark family. All scale
datapoints belonging to that component are included:

```bash
mgym suite dispatch metriq_score_1_0 \
    --component qft --provider ibm --device <device>
```

The option is repeatable when several components are wanted:

```bash
mgym suite dispatch metriq_score_1_0 \
    -c bseq -c wit --provider ibm --device <device>
```

An unknown or repeated component is rejected before provider initialization.

## Run the full suite

The complete suite can consume substantial hardware quota, runtime, and money.
It is therefore never dispatched implicitly. Use `--all` as the explicit
opt-in:

```bash
mgym suite dispatch metriq_score_1_0 \
    --all --provider ibm --device <device>
```

Calling the command without either `--component` or `--all` prints the warning
and available components, then exits before connecting to the provider.

!!! warning
    Check device capabilities and account limits before dispatch. The canonical
    suite includes a 128-qubit Mirror Circuits run and a 100-qubit EPLG chain.
    Its score-compatible CLOPS configuration uses IBM Runtime twirling and a
    session, so it is IBM-specific. No single device or provider is guaranteed
    to support every component.

## Canonical configurations versus examples

Files under `metriq_gym/schemas/examples/` are generic, editable examples for
learning and ad hoc runs. They do not define Metriq Score 1.0 and should not be
used as substitutes for the canonical suite.

To adapt a configuration for a constrained platform, copy the canonical file
and change the copy. Such a run may be useful, but changed parameters or missing
scale datapoints may make its output ineligible for the published score
definition.
