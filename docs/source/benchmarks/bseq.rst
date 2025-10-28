BSEQ (Bell State Effective Qubits)
==================================

Overview
--------

The BSEQ benchmark probes a device's ability to entangle pairs of qubits and detect
violations of the CHSH inequality across the hardware graph. Circuits are generated from a
coloring of the device connectivity map, run with four measurement bases, and scored by how
many connected pairs demonstrate Bell-state behaviour.


Key Resources
-------------

- Metriq profile: https://beta.metriq.info/benchmarks/bseq
- Implementation lineage: Qiskit Device Benchmarking ``bseq`` routines by Paul Nation
- Background reading: J. F. Clauser *et al.*, *Proposed Experiment to Test Local Hidden-Variable
  Theories*, Phys. Rev. Lett. 23, 880 (1969)


Schema Parameters
-----------------

Defined in ``metriq_gym/schemas/bseq.schema.json``.

- ``benchmark_name`` (string, required): must be ``"BSEQ"``.
- ``shots`` (integer, optional, default 1000): measurement repetitions for each circuit. Increase
  to reduce sampling noise, balanced against queue time.

Example payload: ``metriq_gym/schemas/examples/bseq.example.json``.


Running the Benchmark
---------------------

Dispatch from the CLI after syncing dependencies:

.. code-block:: sh

   uv run mgym job dispatch metriq_gym/schemas/examples/bseq.example.json -p local -d aer_simulator

Swap ``-p``/``-d`` for a remote provider and target device as needed. The dispatch step stores the
provider job identifiers and device topology so the poll phase can reconstruct the colouring data.


Result Fields and Interpretation
--------------------------------

Polling returns a ``BSEQResult`` object with two metrics:

- ``largest_connected_size``: size of the largest connected subgraph of qubit pairs that violated
  the CHSH inequality. Higher values indicate that entanglement can be distributed across a larger
  portion of the device.
- ``fraction_connected``: ``largest_connected_size`` divided by the total number of qubits
  discovered during dispatch. This normalises the score for devices of different sizes; values
  closer to 1.0 indicate the entire system produced Bell-state-quality correlations.

Example:

.. code-block:: text

   BSEQResult(largest_connected_size=100, fraction_connected=0.7874)

Use ``fraction_connected`` to compare devices of different scales, while ``largest_connected_size``
is helpful for gauging the absolute reach of entanglement on a specific topology.
