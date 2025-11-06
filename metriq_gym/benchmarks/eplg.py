"""
EPLG benchmark implementation for Metriq‑Gym.

This benchmark uses the Layer Fidelity protocol from Qiskit Experiments to
estimate the error per layered two‑qubit gate (EPLG) on a chain of qubits.
The algorithm creates two disjoint layers of two‑qubit gates along a user
specified list of physical qubits, generates randomized benchmarking circuits
of varying lengths and samples, executes them on the provided quantum device,
and analyses the resulting data to extract the EPLG.  The final figure of
merit reported to the command line interface is the EPLG associated with the
longest chain length specified in the input configuration.  For convenience
the full analysis results are also returned as a pandas DataFrame so that
users can reproduce the Layer Fidelity and related plots on their own.

The benchmark accepts the following parameters (specified in the schema
``eplg.schema.json``):

``qubit_num`` (required): Positive integer specifying the number of qubits
    in the chain on which to run the layered fidelity experiment. A connected
    chain of this length is selected at runtime from the backend coupling map,
    and two disjoint layers of two‑qubit gates are constructed along it.

``lengths`` (required): A list of positive integers specifying the sequence
    lengths (i.e. the number of layers) at which to sample the randomized
    benchmarking circuits.  Larger values correspond to longer circuits and
    provide a better estimate of the asymptotic error rate.

``num_samples`` (required): The number of random circuits to sample for each
    element of ``lengths``.

``nshots`` (required): The number of measurement shots to use when executing
    the circuits on the backend.  A larger number of shots reduces the
    statistical uncertainty in the estimation of fidelities.

``seed`` (optional): An integer seed to initialize the pseudo‑random number
    generator used when generating the randomized benchmarking circuits.  If
    omitted, the generator is seeded randomly.

``two_qubit_gate`` (optional): A string identifying which two‑qubit gate to
    use when constructing the circuit layers.  If omitted, Qiskit will
    select a suitable native gate supported by the target backend.

``one_qubit_basis_gates`` (optional): A list of one‑qubit gate names used to
    synthesize 1‑qubit Cliffords.  If omitted, Qiskit will default to the
    basis gates supported by the backend.

The benchmark returns a single floating point number representing the error
per layered gate at the largest sequence length provided, along with a
pandas DataFrame summarizing intermediate metrics.  This DataFrame contains
rows for chain‑level metrics (layer fidelity and EPLG) as well as
approximate process fidelities for each 1‑qubit and 2‑qubit subsystem of
the selected chain.  Including these additional metrics enables users to
reproduce plots similar to those shown in the accompanying notebook,
highlighting 1Q/2Q process fidelities and their contributions to the chain
fidelity.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import random
import numpy as np
import pandas as pd  # type: ignore
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qbraid import QuantumDevice, QuantumJob, GateModelResultData

from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.qplatform.device import connectivity_graph


def _pick_twoq_gate(backend, prefer=None):
    if prefer and prefer in backend.profile.basis_gates:
        return prefer
    for g in ("ecr", "cx", "cz", "iswap", "rxx", "xx_plus_yy", "xx", "ms"):
        if g in backend.profile.basis_gates:
            return g
    return None


def _allowed_edges(G_und, backend=None, twoq_gate=None, require_gate=True):
    """Return a set of undirected edges {(u,v)} allowed for the chain."""
    edges = [tuple(e) for e in G_und.edge_list()]
    if not require_gate or backend is None:
        return {tuple(sorted(e)) for e in edges}
    gate = _pick_twoq_gate(backend, twoq_gate)
    if gate is None:
        return {tuple(sorted(e)) for e in edges}  # can't verify—allow all couplings
    # print(dict(backend.profile))
    # gmap = backend.target[gate]
    allowed = set()
    for u, v in edges:
        # if (u, v) in gmap or (v, u) in gmap:
        allowed.add(tuple(sorted((u, v))))
    return allowed


def random_chain_fast(
    G, length, *, seed=None, backend=None, twoq_gate=None, require_gate=True, restarts=200
):
    """
    Sample ONE simple path of `length` nodes quickly.
    Tries `restarts` random seeds; each attempt grows from a random edge,
    extending randomly at head/tail. No global enumeration or deep backtracking.
    """
    rng = random.Random(seed)
    G_und = G.to_directed()

    allowed = _allowed_edges(G_und, backend, twoq_gate, require_gate)
    if not allowed:
        raise RuntimeError("No allowed 2-qubit edges found to form a chain.")

    # Build adjacency from allowed edges only
    n_nodes = G_und.num_nodes()
    adj = {i: [] for i in range(n_nodes)}
    for u, v in allowed:
        adj[u].append(v)
        adj[v].append(u)

    # Edge list for random starts
    allowed_edges_list = list(allowed)

    for _ in range(restarts):
        # Start from a random allowed edge: path of length 2 already
        u, v = rng.choice(allowed_edges_list)
        path = [u, v]
        used = {u, v}

        # Grow until target length
        while len(path) < length:
            extended = False
            # Try to extend either end (random order)
            for side in rng.sample(("head", "tail"), 2):
                if side == "head":
                    cur = path[0]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.insert(0, w)
                        used.add(w)
                        extended = True
                        break
                else:
                    cur = path[-1]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.append(w)
                        used.add(w)
                        extended = True
                        break
            if not extended:
                # Dead end – restart with a new random edge
                break

        if len(path) == length:
            return path

    raise RuntimeError(
        f"Failed to sample a chain of length={length} after {restarts} restarts. "
        "Try increasing `restarts`, lowering `length`, or setting `require_gate=False`."
    )


@dataclass
class EPLGData(BenchmarkData):
    """Container for intermediate EPLG metadata.

    Attributes:
        physical_qubit_num: int
            The number of physical qubits in the selected chain.
    """

    physical_qubit_num: int


class EPLGResult(BenchmarkResult):
    """Results of the EPLG benchmark.

    Attributes:
        eplg: Error per layered gate evaluated at the longest chain length.
    """

    eplg: float


class EPLG(Benchmark[EPLGData, EPLGResult]):
    """Benchmark class for the EPLG (Layer Fidelity) experiment."""

    def dispatch_handler(self, device: QuantumDevice) -> EPLGData:
        """Generate and submit layer fidelity circuits to the quantum device.

        The dispatch handler is responsible for sampling a random chain from the
        backend coupling map, constructing the two disjoint layers required for
        the layer fidelity experiment, generating randomized benchmarking
        circuits via :class:`~qiskit_experiments.library.randomized_benchmarking.LayerFidelity`,
        and submitting them to the provided device.  We use
        :meth:`~metriq_gym.benchmarks.benchmark.BenchmarkData.from_quantum_job` to
        capture the provider job IDs along with all relevant parameters.  No
        analysis is performed at this stage; results are processed in
        :meth:`poll_handler` once the jobs complete.

        Args:
            device: The quantum device to run the circuits on.

        Returns:
            An :class:`EPLGData` object capturing the dispatched job IDs and
            experiment parameters.
        """
        # Extract parameters from the pydantic model stored on the benchmark.
        params = self.params
        qubit_num: int = params.qubit_num
        lengths: Sequence[int] = params.lengths
        num_samples: int = params.num_samples
        nshots: int = params.nshots
        seed: Optional[int] = getattr(params, "seed", None)
        two_qubit_gate: Optional[str] = getattr(params, "two_qubit_gate", None)
        one_qubit_basis_gates: Optional[Sequence[str]] = getattr(
            params, "one_qubit_basis_gates", None
        )

        # Validate requested gates against the backend.  Raise an error if the
        # user specifies a 2‑qubit or 1‑qubit gate not supported on the target.
        supported_ops: Optional[set] = None
        try:
            supported_ops = device.profile.basis_gates
        except Exception:
            pass
        if two_qubit_gate and supported_ops and two_qubit_gate not in supported_ops:
            raise ValueError(f"Two‑qubit gate '{two_qubit_gate}' is not supported by the backend.")
        if one_qubit_basis_gates and supported_ops:
            for g in one_qubit_basis_gates:
                if g not in supported_ops:
                    raise ValueError(f"One‑qubit gate '{g}' is not supported by the backend.")

        graph = connectivity_graph(device)
        selected_chain = random_chain_fast(
            graph,
            qubit_num,
            seed=seed,
            backend=device,
            twoq_gate=two_qubit_gate,  # or "ecr"/"cx"/...
            require_gate=True,  # set False to ignore calibration availability
            restarts=500,  # bump if your graph is sparse or the chain is long
        )

        # Decompose the selected chain into two disjoint layers by pairing adjacent
        # qubits.  The first layer pairs (q0,q1),(q2,q3),... and the second layer
        # pairs (q1,q2),(q3,q4),... as required by the Layer Fidelity protocol.
        edges = list(zip(selected_chain[:-1], selected_chain[1:]))
        two_qubit_layers: List[List[Tuple[int, int]]] = [edges[0::2], edges[1::2]]

        # Construct keyword arguments for the LayerFidelity experiment.  We set
        # ``backend=None`` here since circuits are executed via the qBraid device
        # rather than through Qiskit Experiments.  Optional parameters are only
        # included if specified by the user.
        lf_kwargs: dict = {
            "physical_qubits": selected_chain,
            "two_qubit_layers": two_qubit_layers,
            "lengths": lengths,
            "num_samples": num_samples,
            "seed": seed,
            "backend": device,  # CL TODO: this seems not working and one still need input 1Q/2Q gate names
        }

        # Alternatively:
        # If no two_qubit_gate: use backend.profile.basis_gates and find the two_qubit_gate/one_qubit_basis_gates
        if two_qubit_gate:
            lf_kwargs["two_qubit_gate"] = two_qubit_gate
        if one_qubit_basis_gates:
            lf_kwargs["one_qubit_basis_gates"] = tuple(one_qubit_basis_gates)

        experiment = LayerFidelity(**lf_kwargs)

        # Limit the number of circuits per job to a conservative default to
        # mitigate large payloads on some providers.
        try:
            experiment.set_experiment_options(max_circuits=144)
        except Exception:
            pass
        # Generate all circuits for the experiment.  The ordering of circuits
        # matches that expected by the analysis routine: the first
        # ``num_samples`` circuits correspond to ``lengths[0]``, the next
        # ``num_samples`` correspond to ``lengths[1]``, and so forth.
        circuits = experiment._transpiled_circuits()
        device.set_options(transform=False)

        # Submit the circuits to the device and capture the resulting QuantumJob.
        quantum_job = device.run(circuits, shots=nshots)

        # Package the job and experimental parameters into an EPLGData instance.
        return EPLGData.from_quantum_job(
            quantum_job,
            physical_qubit_num=max(selected_chain) + 1,  # qubit indices are zero-based
        )

    def poll_handler(
        self,
        job_data: EPLGData,
        result_data: List[GateModelResultData],
        quantum_jobs: List[QuantumJob],
    ) -> EPLGResult:
        """Compute the EPLG from completed job results.

        In the polling phase we process the results returned by the device
        and estimate the error per layered gate at each sequence length.  The
        randomized benchmarking protocol is such that the correct output
        bitstring for each circuit is the all‑zero string (in the logical
        ordering of the selected chain).  We compute the survival probability
        for each circuit as the fraction of measurement shots yielding this
        bitstring, average these values across the ``num_samples`` circuits
        corresponding to each sequence length, and treat the average survival
        probability as the chain fidelity ``F(L)`` for that length ``L``.  The
        error per layered gate at length ``L`` is then extracted via

        EPLG = 1 - F(L)^{1/(N_{2q}(L))}

        where `N_{2q}(L)` is the total number of two‑qubit gates in a
        single circuit of length :math:`L`.  For a chain of ``n`` qubits the
        two disjoint layers together apply ``n-1`` two‑qubit gates per layer
        pair, so a sequence of length :math:`L` contains :math:`N_{2q}(L) = (n-1)/L`
        gates.  We return the EPLG associated with the largest sequence length
        provided in the input, along with a DataFrame summarizing the chain
        fidelities, EPLG values, and approximate 1Q/2Q process fidelities for
        each length.

        Args:
            job_data: Parameters captured at dispatch time, including the
                physical qubits, sequence lengths, number of samples and shots.
            result_data: A list of :class:`GateModelResultData` objects
                containing measurement counts for each circuit.  The ordering of
                this list matches the ordering of circuits returned by
                :meth:`LayerFidelity.circuits`.
            quantum_jobs: The underlying QuantumJob objects corresponding to the
                dispatched circuits (unused here but required by the interface).

        Returns:
            An :class:`EPLGResult` containing the EPLG at the largest sequence
            length and a pandas DataFrame summarizing the analysis results.  Each
            row of the DataFrame has columns ``name``, ``value``, ``qubits`` and
            ``length``.  Rows with ``name`` equal to ``"LF"`` and ``"EPLG"``
            record the average chain fidelity and error per layered gate at each
            sequence length, with ``qubits`` listing the entire selected chain.
            Rows with ``name`` equal to ``"ProcessFidelity"`` record the
            approximate process fidelity for individual 1‑qubit and 2‑qubit
            subsystems of the chain; here ``qubits`` identifies the physical
            qubit or qubit pair and ``value`` is the average probability of
            returning to the ground state ("00" for 2Q).  These additional
            results enable users to replicate analyses similar to the notebook
            showing 1Q/2Q process fidelities.
        """
        params = self.params
        n_qubits: int = params.qubit_num
        physical_qubit_num: int = job_data.physical_qubit_num
        lengths: List[int] = params.lengths
        num_samples: int = params.num_samples

        # TODO: check qiskit analysis_results for data processing
        counts_list = flatten_counts(result_data)
        # Validate the number of result entries matches the number of circuits.
        expected_circuits = len(lengths) * num_samples * 2
        if len(counts_list) != expected_circuits:
            raise RuntimeError(
                f"Expected {expected_circuits} result data entries but got {len(counts_list)}."
            )

        # Define the expected zero bitstring; this corresponds to the ideal
        # outcome for direct randomized benchmarking circuits.  We assume the
        # qubits map to the bitstring ordering used by the device; if not, the
        # zero string will still be present as some key.
        zero_string = "0" * physical_qubit_num

        # Compute average chain fidelity F(L) for each sequence length.  The
        # ordering of circuits is such that the first ``num_samples`` entries
        # correspond to lengths[0], the next ``num_samples`` to lengths[1], etc.
        chain_fids: List[float] = []
        idx = 0
        for L in lengths:
            subset = counts_list[idx : idx + num_samples]
            idx += num_samples
            surv_probs: List[float] = []
            for counts in subset:
                total_shots = sum(counts.values()) if hasattr(counts, "values") else 0
                prob0 = 0.0
                if total_shots > 0:
                    # direct lookup for the all-zero string; some backends may use
                    # integer keys or unpadded strings, so we fall back gracefully.
                    if zero_string in counts:
                        prob0 = counts.get(zero_string, 0) / total_shots
                    elif "0" in counts:
                        prob0 = counts.get("0", 0) / total_shots
                    else:
                        prob0 = 0.0
                surv_probs.append(prob0)
            chain_fids.append(float(np.mean(surv_probs)) if surv_probs else 0.0)

        if not chain_fids:
            raise RuntimeError("No survival probabilities computed; cannot estimate EPLG.")

        # Compute the number of two‑qubit gates for each sequence length.  In
        # Layer Fidelity each pair of disjoint layers applies exactly ``n_qubits-1``
        # two‑qubit gates across the chain, so a sequence of length L uses
        # ``(n_qubits-1) * L`` such gates.  Guard against the degenerate case
        # where ``n_qubits`` is 1.
        n2q_per_length: List[int] = []
        for L in lengths:
            num2q = max(1, (n_qubits - 1) * L)
            n2q_per_length.append(num2q)

        # Compute EPLG values for each length using the chain fidelity and the
        # corresponding total number of two‑qubit gates.  Values are clipped to
        # lie between 0 and 1 for numerical stability.
        chain_eplgs: List[float] = []
        for fid, num2q in zip(chain_fids, n2q_per_length):
            f = max(0.0, min(1.0, fid))
            eplg = 1.0 - (f ** (1.0 / num2q))
            chain_eplgs.append(eplg)

        # In addition to chain‑level metrics, estimate the process fidelity for
        # each single‑qubit and two‑qubit subsystem by marginalizing counts.  For
        # a 1Q subsystem corresponding to the `i`‑th qubit in the selected chain
        # we compute the probability that this qubit is measured in the 0 state.
        # For a 2Q subsystem defined by adjacent qubits `(i, i+1)` we compute the
        # probability that both qubits are measured in the 00 state.  These
        # approximate process fidelities are averaged over the samples at each
        # sequence length.

        # Precompute lists of subsystem indices for 1Q and 2Q process fidelities.
        one_q_indices = list(range(n_qubits))
        two_q_indices = [(i, i + 1) for i in range(max(0, n_qubits - 1))]

        # Containers for process fidelities per length.
        pf_1q_by_length: List[dict] = []
        pf_2q_by_length: List[dict] = []

        idx_counts = 0
        for L in lengths:
            subset = counts_list[idx_counts : idx_counts + num_samples]
            idx_counts += num_samples
            # One‑qubit process fidelities
            pf1_dict: dict = {}
            for qpos in one_q_indices:
                surv_probs_q: List[float] = []
                for counts in subset:
                    total_shots = sum(counts.values()) if hasattr(counts, "values") else 0
                    prob0 = 0.0
                    if total_shots > 0:
                        count_zero = 0
                        for key, cnt in counts.items():
                            # Convert the key to a bitstring.  Keys may be strings or integers.
                            if isinstance(key, str):
                                bitstr = key
                            else:
                                # interpret integer as binary string with no '0b' prefix
                                bitstr = format(key, "b")
                            # pad to the full chain length
                            bitstr = bitstr.zfill(n_qubits)
                            # We treat the reversed bitstring so that index corresponds to
                            # the position in the selected chain.  This is an assumption
                            # about bit ordering and may differ on some backends.
                            rev = bitstr[::-1]
                            if qpos < len(rev) and rev[qpos] == "0":
                                count_zero += cnt
                        prob0 = count_zero / total_shots
                    surv_probs_q.append(prob0)
                pf1_dict[qpos] = float(np.mean(surv_probs_q)) if surv_probs_q else 0.0
            pf_1q_by_length.append(pf1_dict)

            # Two‑qubit process fidelities
            pf2_dict: dict = {}
            for i, j in two_q_indices:
                surv_probs_pair: List[float] = []
                for counts in subset:
                    total_shots = sum(counts.values()) if hasattr(counts, "values") else 0
                    prob00 = 0.0
                    if total_shots > 0:
                        count_zero2 = 0
                        for key, cnt in counts.items():
                            if isinstance(key, str):
                                bitstr = key
                            else:
                                bitstr = format(key, "b")
                            bitstr = bitstr.zfill(n_qubits)
                            rev = bitstr[::-1]
                            # check both qubits are 0
                            if i < len(rev) and j < len(rev):
                                if rev[i] == "0" and rev[j] == "0":
                                    count_zero2 += cnt
                        prob00 = count_zero2 / total_shots
                    surv_probs_pair.append(prob00)
                pf2_dict[(i, j)] = float(np.mean(surv_probs_pair)) if surv_probs_pair else 0.0
            pf_2q_by_length.append(pf2_dict)

        # Build a comprehensive DataFrame summarizing the metrics for each length.
        # Each row corresponds to a particular metric and subsystem at a specific
        # sequence length.  The columns ``name``, ``value``, ``qubits`` and
        # ``length`` allow filtering similar to the Qiskit analysis DataFrame.
        rows = []
        for idx_L, L in enumerate(lengths):
            # Chain‑level fidelity and EPLG
            rows.append(
                {
                    "name": "LF",
                    "value": chain_fids[idx_L],
                    # "qubits": tuple(chain_qubits), # TODO: add
                    "length": L,
                }
            )
            rows.append(
                {
                    "name": "EPLG",
                    "value": chain_eplgs[idx_L],
                    # "qubits": tuple(chain_qubits),
                    "length": L,
                }
            )
            # One‑qubit process fidelities
            for qpos, pf_val in pf_1q_by_length[idx_L].items():
                rows.append(
                    {
                        "name": "ProcessFidelity",
                        "value": pf_val,
                        # "qubits": (chain_qubits[qpos],),
                        "length": L,
                    }
                )
            # Two‑qubit process fidelities
            for (i, j), pf_val in pf_2q_by_length[idx_L].items():
                rows.append(
                    {
                        "name": "ProcessFidelity",
                        "value": pf_val,
                        # "qubits": (chain_qubits[i], chain_qubits[j]),
                        "length": L,
                    }
                )
        _ = pd.DataFrame(rows)

        # The benchmark’s single scalar output is the EPLG at the largest length.
        final_eplg = chain_eplgs[-1]
        return EPLGResult(eplg=final_eplg)
