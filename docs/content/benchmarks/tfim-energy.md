# TFIM Energy

The **TFIM Energy** benchmark evaluates how accurately a quantum execution stack
estimates the energy of a fixed four-qubit transverse-field Ising model
workload.

## Hamiltonian

The benchmark uses the open-chain Hamiltonian:

```text
H = -J sum_i Z_i Z_(i+1) - h sum_i X_i
```

with `J = 1`, `h = 1`, and four qubits.

## State preparation

The workload uses a fixed, reproducible two-layer variational ansatz. The
circuit starts in `|+>^4` and applies two layers consisting of:

1. one `RY` rotation per qubit; and
2. a nearest-neighbour CNOT chain.

The fixed parameter vector is:

```text
[
  0.292792927,
 -0.0000282104047,
 -0.100168525,
 -0.100168480,
 -0.292808429,
 -0.518628570,
 -0.562315096,
 -0.518620523
]
```

## Measurement workload

The Hamiltonian contains seven Pauli terms:

```text
Z0Z1
Z1Z2
Z2Z3
X0
X1
X2
X3
```

One circuit is submitted per term.

- `ZZ` terms are measured in the computational basis.
- `X` terms are rotated with a Hadamard before measurement.
- Every term receives the same configured shot count.

The seven circuits are submitted as one batch where the provider supports
batching.

## Reference values

The implementation computes both reference energies from the benchmark
definition.

| Quantity | Value |
|---|---:|
| Exact ground-state energy | -4.7587704831 |
| Ideal variational energy | -4.7575478600 |
| Intrinsic ansatz energy error | 0.0012226231 |

## Metrics

The measured energy is reconstructed as:

```text
E_measured = sum_k c_k <P_k>
```

For a Pauli-product expectation with outcomes `+1` and `-1`, the finite-shot
standard error is:

```text
sigma_k = sqrt((1 - <P_k>^2) / N_k)
```

Assuming the Hamiltonian terms are sampled independently, energy uncertainty is:

```text
sigma_E = sqrt(sum_k c_k^2 sigma_k^2)
```

### Execution energy error

```text
epsilon_exec = |E_measured - E_ideal_ansatz|
```

This isolates execution effects from the ansatz approximation error.

### Application energy error

```text
epsilon_app = |E_measured - E_ground|
```

This is the end-to-end application error.

### Proposed normalized score

The proposed scalar score is:

```text
score = max(0, min(1, 1 - epsilon_app / |E_ground|))
```

The score definition remains part of the upstream design discussion. Raw energy
and error metrics are reported independently.

## Why two reference energies?

A measured energy can differ from the exact ground-state energy because the
fixed ansatz is approximate and because the execution stack introduces
sampling, compilation, noise, and hardware effects.

Reporting both execution and application errors prevents the ansatz
approximation from being attributed entirely to the hardware.

## Error mitigation

The canonical TFIM Energy benchmark is **unmitigated**. Error mitigation is a
separate companion analysis so raw provider and hardware results remain
comparable.

## Configuration

```json
{
  "benchmark_name": "TFIM Energy",
  "num_qubits": 4,
  "coupling_j": 1.0,
  "field_h": 1.0,
  "shots": 8192
}
```

## Local execution

```bash
mgym job dispatch metriq_gym/schemas/examples/tfim_energy.example.json \
  --provider local --device aer_simulator

mgym job poll latest
```

## MVP scope

The first version fixes the number of qubits, Hamiltonian coefficients, ansatz
architecture, and ansatz parameters. Future versions can add scale points after
the initial workload and result model are validated.
