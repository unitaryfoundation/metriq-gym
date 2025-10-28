WIT (Wormhole-Inspired Teleportation)
====================================

Overview
--------

The WIT benchmark prepares a seven- or six-qubit circuit inspired by holographic wormhole
teleportation protocols. It measures the expectation value of a Pauli-Z observable after
performing a sequence of entangling operations that emulate the dynamics discussed in
Shapoval *et al.* (2023). Metriq Gym follows the reference implementation contributed by
Paul Nation for IBM Quantum hardware and adapts it for provider-agnostic execution.


Key Resources
-------------

- Metriq profile: https://beta.metriq.info/benchmarks/wit
- Reference paper: I. Shapoval *et al.*, *Towards Quantum Gravity in the Lab on Quantum Processors*,
  Quantum 7, 1138 (2023), https://arxiv.org/abs/2205.14081
- Companion software: https://gitlab.com/ishapova/qglab/-/blob/master/scripts/wormhole.py


Schema Parameters
-----------------

Defined in ``metriq_gym/schemas/wit.schema.json``.

- ``benchmark_name`` (string, required): must be ``"WIT"``.
- ``shots`` (integer, optional, default 1000): number of measurement repetitions. Higher shot
  counts reduce the statistical uncertainty on the reported expectation value.
- ``num_qubits`` (integer, optional, default 6): supported values are ``6`` and ``7``. Choose ``7``
  to match the full circuit shown in Fig. 4 of the reference paper when the device topology allows
  it.

Example payload: ``metriq_gym/schemas/examples/wit.example.json``.


Running the Benchmark
---------------------

Dispatch the benchmark with your preferred backend:

.. code-block:: sh

   uv run mgym job dispatch metriq_gym/schemas/examples/wit.example.json -p local -d aer_simulator

Update ``num_qubits`` in the payload to match the hardware constraint before dispatching to a real
device. During polling, Metriq Gym aggregates the observed counts and computes the expectation
value and uncertainty.


Result Fields and Interpretation
--------------------------------

Polling returns a ``WITResult`` object with one metric:

- ``expectation_value``: stored as a ``BenchmarkScore`` containing ``value`` and ``uncertainty``.
  The value is the single-qubit Pauli-Z expectation estimated from the measurement shots, and the
  uncertainty is a binomial standard deviation computed from the observed counts.

Interpretation guidelines:

- Ideal teleportation yields a value close to ``+1``; lower values indicate noise or imperfect
  parameter settings.
- Compare the reported uncertainty with the expectation to decide whether additional shots are
  warranted.
- When running across devices, ensure you use the same ``num_qubits`` choice to make results
  comparable.
