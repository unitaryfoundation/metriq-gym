# QED-C Benchmarks

The QED-C (Quantum Economic Development Consortium) benchmarks are a suite of application-oriented quantum algorithms for evaluating quantum computers.

## Overview

These benchmarks implement standard quantum algorithms and evaluate device performance across varying problem sizes. They're based on the QED-C Application-Oriented Benchmarks suite.

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [Bernstein-Vazirani](#bernstein-vazirani) | Hidden string discovery |
| [Phase Estimation](#phase-estimation) | Eigenvalue estimation |
| [Hidden Shift](#hidden-shift) | Function shift discovery |
| [Quantum Fourier Transform](#quantum-fourier-transform) | QFT implementation |

---

## Bernstein-Vazirani

Finds a hidden binary string encoded in a function using a single query.

### Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Bernstein-Vazirani"` |
| `shots` | integer | No | 1000 | Measurement shots |
| `min_qubits` | integer | No | 2 | Starting qubit count |
| `max_qubits` | integer | No | 6 | Maximum qubit count |
| `skip_qubits` | integer | No | 1 | Step size between sizes |
| `max_circuits` | integer | No | 3 | Circuits per qubit count |
| `method` | integer | No | 1 | QED-C method variant (1 or 2) |
| `input_value` | integer | No | - | Specific secret string |

### Example

```json
{
  "benchmark_name": "Bernstein-Vazirani",
  "min_qubits": 4,
  "max_qubits": 8,
  "shots": 5000
}
```

---

## Phase Estimation

Estimates the eigenvalue of a unitary operator.

### Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Phase Estimation"` |
| `shots` | integer | No | 1000 | Measurement shots |
| `min_qubits` | integer | No | 3 | Starting qubit count |
| `max_qubits` | integer | No | 6 | Maximum qubit count |
| `skip_qubits` | integer | No | 1 | Step size between sizes |
| `max_circuits` | integer | No | 3 | Circuits per qubit count |
| `init_phase` | float | No | - | Initial phase theta |
| `use_midcircuit_measurement` | boolean | No | false | Enable mid-circuit measurements |

### Example

```json
{
  "benchmark_name": "Phase Estimation",
  "min_qubits": 4,
  "max_qubits": 7,
  "shots": 5000,
  "use_midcircuit_measurement": false
}
```

---

## Hidden Shift

Discovers a hidden shift in bent Boolean functions.

### Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Hidden Shift"` |
| `shots` | integer | No | 1000 | Measurement shots |
| `min_qubits` | integer | No | 2 | Starting qubit count |
| `max_qubits` | integer | No | 6 | Maximum qubit count |
| `skip_qubits` | integer | No | 1 | Step size between sizes |
| `max_circuits` | integer | No | 3 | Circuits per qubit count |
| `input_value` | integer | No | - | Specific secret value |

### Example

```json
{
  "benchmark_name": "Hidden Shift",
  "min_qubits": 4,
  "max_qubits": 8,
  "shots": 5000
}
```

---

## Quantum Fourier Transform

Implements the Quantum Fourier Transform, a key subroutine in many quantum algorithms.

### Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `benchmark_name` | string | Yes | - | Must be `"Quantum Fourier Transform"` |
| `shots` | integer | No | 1000 | Measurement shots |
| `min_qubits` | integer | No | 2 | Starting qubit count |
| `max_qubits` | integer | No | 6 | Maximum qubit count |
| `skip_qubits` | integer | No | 1 | Step size between sizes |
| `max_circuits` | integer | No | 3 | Circuits per qubit count |
| `method` | integer | No | 1 | Method variant (1 or 2) |
| `input_value` | integer | No | - | Specific input state |
| `use_midcircuit_measurement` | boolean | No | false | Enable mid-circuit measurements |

### Example

```json
{
  "benchmark_name": "Quantum Fourier Transform",
  "min_qubits": 3,
  "max_qubits": 8,
  "shots": 5000,
  "method": 1
}
```

---

## Usage

All QED-C benchmarks follow the same dispatch pattern:

```bash
# Dispatch
mgym job dispatch metriq_gym/schemas/examples/bernstein_vazirani.example.json \
    --provider ibm --device ibm_sherbrooke

# Poll results
mgym job poll <JOB_ID>
```

## Results Format

QED-C benchmarks return:

| Metric | Description |
|--------|-------------|
| `success_rate` | Fraction of correct answers |
| `fidelity` | Circuit execution fidelity |
| `qubit_scaling` | Performance vs. qubit count |

## Scaling Analysis

These benchmarks are designed to test scaling:

```json
{
  "benchmark_name": "Bernstein-Vazirani",
  "min_qubits": 4,
  "max_qubits": 20,
  "skip_qubits": 2
}
```

This runs at 4, 6, 8, 10, 12, 14, 16, 18, and 20 qubits.

## Method Variants

Some benchmarks support multiple implementation methods:
- **Method 1**: Standard textbook implementation
- **Method 2**: Alternative or optimized implementation

## Mid-Circuit Measurements

Phase Estimation and QFT support iterative versions with mid-circuit measurements:

```json
{
  "benchmark_name": "Phase Estimation",
  "use_midcircuit_measurement": true
}
```

!!! note
    Mid-circuit measurements require hardware support. Check provider documentation.

## References

- [QED-C Application-Oriented Benchmarks](https://github.com/SRI-International/QC-App-Oriented-Benchmarks)
- [arXiv:2110.03137](https://arxiv.org/abs/2110.03137) - Application-Oriented Performance Benchmarks
