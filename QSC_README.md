# QSC (Quantum Simulation Circuits) Benchmark

This is a **private** fork of [metriq-gym](https://github.com/unitaryfoundation/metriq-gym) for developing the QSC benchmark integration.

The QSC benchmark implements quantum dynamical simulation via Hamiltonian time evolution using Trotter decomposition. It is based on the [QSC-Applications](https://github.com/vprusso/QSC-Applications) project and the methods described in [arXiv:1912.06076](https://arxiv.org/abs/1912.06076).

## Status

- **Inputs**: Well-defined (see below)
- **Outputs**: Preliminary (per-qubit Pauli expectation values); output format will be refined as the QSC algorithm is finalized
- **Branch**: `feat/qsc-benchmark`

## Setup

```bash
git clone https://github.com/vprusso/metriq-gym-qsc.git
cd metriq-gym-qsc
git submodule update --init --recursive
git checkout feat/qsc-benchmark
uv sync
uv pip install -e .
```

## Running the Benchmark

### 1. Dispatch on the local simulator

```bash
uv run mgym job dispatch metriq_gym/schemas/examples/qsc.example.json -p local -d aer_simulator
```

### 2. Poll for results (immediate for local simulator)

```bash
uv run mgym job poll latest
```

### Example output

```
results:
  expectation_values:
    qubit 0: 0.827 +/- 0.012
    qubit 1: 0.815 +/- 0.012
    qubit 2: 0.531 +/- 0.016
    qubit 3: 0.669 +/- 0.015
    qubit 4: 0.914 +/- 0.009
  pauli_basis: X
  score: 0.751 +/- 0.013
```

### 3. Estimate resource costs (optional)

```bash
uv run mgym job estimate metriq_gym/schemas/examples/qsc.example.json -p local -d aer_simulator
```

```
| Metric                |   Value |
|-----------------------|---------|
| Jobs                  |       1 |
| Circuits              |       1 |
| Total shots           |   1_000 |
| Max qubits            |       5 |
| Total 1q gates        |     196 |
| Total 2q gates        |      74 |
| Circuit depth         |     193 |
| Total HQCs            |   207.2 |
```

## Benchmark Inputs

All parameters are configured via a JSON file. See [`metriq_gym/schemas/examples/qsc.example.json`](metriq_gym/schemas/examples/qsc.example.json) for a complete example.

| Parameter        | Type    | Default | Description                                          |
|------------------|---------|---------|------------------------------------------------------|
| `benchmark_name` | string  | —       | Must be `"QSC"` (required)                           |
| `num_qubits`     | integer | 5       | Number of qubits / lattice sites (min: 2)            |
| `delta`          | number  | 0.25    | Anisotropy parameter for ZZ coupling                 |
| `j_coupling`     | number  | 0.23    | Exchange coupling strength (XX, YY, ZZ interactions)  |
| `hx`             | number  | 0.5     | Transverse field strength                            |
| `trotter_order`  | integer | 2       | Suzuki-Trotter decomposition order                   |
| `trotter_steps`  | integer | 1       | Number of Trotter repetitions                        |
| `evolution_time` | number  | 2.0     | Duration of Hamiltonian time evolution               |
| `pauli_basis`    | string  | `"X"`   | Measurement/perturbation basis: `"X"`, `"Y"`, or `"Z"` |
| `shots`          | integer | 1000    | Number of measurement shots                          |

These correspond to the inputs described by Elijah: evolution time, a quantum Hamiltonian (parameterized by `delta`, `j_coupling`, `hx`, `num_qubits`), max depth circuit cost (`trotter_order`, `trotter_steps`), and a Pauli basis.

## Benchmark Outputs (Preliminary)

Currently returns per-qubit expectation values in the specified Pauli basis, each with a standard error (binomial uncertainty). An overall score is computed as the mean across qubits.

The final output format is still being determined and may evolve to include:
- Multiple Pauli bases per run (X, Y, Z arrays)
- 2D arrays of real numbers per basis
- Additional post-processing metrics

## Files Added/Modified

```
metriq_gym/
  benchmarks/qsc.py                  # Benchmark implementation
  schemas/qsc.schema.json            # Input parameter schema
  schemas/examples/qsc.example.json  # Example configuration
  constants.py                       # Added JobType.QSC
  registry.py                        # Registered QSC handlers
```

## Syncing with upstream metriq-gym

```bash
git fetch upstream
git merge upstream/main
```

## References

- [Childs, Maslov, Nam, Ross, Su (2019). "Toward the first quantum simulation with quantum speedup"](https://arxiv.org/abs/1912.06076)
- [metriq-gym: Adding Benchmarks tutorial](https://unitaryfoundation.github.io/metriq-gym/development/adding-benchmarks/)
