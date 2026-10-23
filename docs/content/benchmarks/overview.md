# Benchmarks

Metriq-Gym provides a comprehensive suite of quantum benchmarks to characterize and compare quantum hardware performance.

## Running Benchmarks

### Single Benchmark

```bash
mgym job dispatch metriq_gym/schemas/examples/quantum_volume.example.json \
    --provider local --device aer_simulator
```

### Poll for Results

```bash
mgym job poll <JOB_ID>
```

## Configuration

All benchmarks use JSON configuration files:

- **Schemas**: `metriq_gym/schemas/*.schema.json` - Define parameters, types, and allowed values
- **Examples**: `metriq_gym/schemas/examples/*.example.json` - Generic, editable
  configurations for learning and ad hoc runs
- **Canonical score suite**: `metriq_gym/suites/metriq_score_1_0.json` - Exact
  configurations and scale datapoints for [Metriq Score 1.0](../suites/metriq-score-1.0.md)

Example files are not canonical Metriq Score configurations.

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [Mirror Circuits](#metriq_gym.benchmarks.mirror_circuits) | Tests state fidelity via forward/reverse Clifford layers |
| [EPLG](#metriq_gym.benchmarks.eplg) | Error per layered gate across qubit chains |
| [BSEQ](#metriq_gym.benchmarks.bseq) | Bell state effective qubits via CHSH violation |
| [WIT](#metriq_gym.benchmarks.wit) | Wormhole-inspired teleportation protocol |
| [LR-QAOA](#metriq_gym.benchmarks.lr_qaoa) | Linear-ramp QAOA for Max-Cut optimization |
| [QML Kernel](#metriq_gym.benchmarks.qml_kernel) | Quantum machine learning kernel accuracy |
| [QED-C Benchmarks](#metriq_gym.benchmarks.qedc_benchmarks) | Application-oriented benchmarks (BV, QFT, etc.) |
| [CLOPS](#metriq_gym.benchmarks.clops) | Circuit layer operations per second to measure device speed |
| [GHZ](#metriq_gym.benchmarks.ghz) | GHZ state fidelity, at a fixed size or by searching for the largest one the device can build |

### GHZ results and scoring

The logarithmic-depth preparation follows the approach illustrated by
[Quantinuum's GHZ state fidelity example](https://docs.quantinuum.com/systems/trainings/knowledge_articles/Quantinuum_n56_ghz_state_fidelity.html).
That example combines a population measurement with `N` deterministic equatorial
parity settings to estimate fidelity against GHZ+. Here DFE samples stabilizers
for the same target, while the oscillation methods recover coherence magnitude
and optimize over the relative phase.

Use `num_qubits` to measure one size, or `size_search: true` to test a halving
grid that includes `min_qubits`. Size search reports `largest_passing_size`,
`device_fraction`, and the fidelity and uncertainty at each tested size. The
passing margin is `fidelity - confidence_sigma * uncertainty > fidelity_threshold`.
This is a margin based on estimated standard errors, not a rigorous confidence
guarantee. The search reports the largest **tested** passing size.

For Metriq Score integration, select `largest_passing_size` explicitly for a
size-search component, or `fidelity` with a fixed-size selector. The generic
`score` has different units in these modes. GHZ is not yet part of the canonical
score suite.

DFE estimates fidelity against GHZ+, so a relative phase error reduces its score.
Parity oscillation and compressed sensing estimate fidelity optimized over that
phase; the two-setting method gives a lower bound on this optimized fidelity.
Compare runs using the same method. Compressed sensing assumes a sparse spectrum;
its uncertainty propagates shot noise through the recovered model, and its
`recovered_frequency` is a diagnostic rather than an entanglement certificate.
If the recovered spectrum needs more phase samples, increase `num_phases` or use
uniform parity oscillation.

---

::: metriq_gym.benchmarks.mirror_circuits
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.eplg
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.bseq
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.wit
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.lr_qaoa
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.qml_kernel
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.qedc_benchmarks
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.clops
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3

::: metriq_gym.benchmarks.ghz
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      heading_level: 3
## Adding Custom Benchmarks

See [Adding New Benchmarks](../development/adding-benchmarks.md) to contribute new benchmarks.
