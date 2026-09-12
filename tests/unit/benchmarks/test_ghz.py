from argparse import Namespace
import itertools

import numpy as np
import pytest
import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.ghz import (
    GHZBenchmark,
    bisection_sizes,
    two_setting_sampling_is_adequate,
    GHZResult,
    _bfs_edges,
    build_ghz_circuits,
    estimate_fidelity_compressed_sensing,
    estimate_fidelity_dfe,
    estimate_fidelity_oscillation,
    estimate_fidelity_two_setting_bound,
    sample_ghz_stabilizers,
)
from metriq_gym.benchmarks.benchmark import BenchmarkScore


class TestBfsEdges:
    def test_linear_graph(self):
        graph = rx.generators.path_graph(5)
        edges = _bfs_edges(graph, root=0, num_qubits=5)
        assert len(edges) == 4
        # BFS from 0 on a path: 0-1, 1-2, 2-3, 3-4
        assert edges == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_complete_graph(self):
        graph = rx.generators.complete_graph(6)
        edges = _bfs_edges(graph, root=0, num_qubits=6)
        assert len(edges) == 5
        # A dissemination tree doubles the prepared set each round instead of
        # driving every CNOT from qubit 0, so the controls are not all the root.
        assert len({ctrl for ctrl, _ in edges}) > 1
        # Every target is newly prepared exactly once.
        targets = [t for _, t in edges]
        assert len(set(targets)) == len(targets)

    def test_partial_qubits(self):
        graph = rx.generators.path_graph(10)
        edges = _bfs_edges(graph, root=0, num_qubits=4)
        assert len(edges) == 3
        # Should only use first 4 nodes
        all_nodes = {0}
        for c, t in edges:
            all_nodes.add(c)
            all_nodes.add(t)
        assert len(all_nodes) == 4


class TestBuildGhzCircuits:
    def test_two_setting_bound_returns_two_circuits(self):
        graph = rx.generators.complete_graph(4)
        circuits, data_qubits = build_ghz_circuits(graph, num_qubits=4, method="two_setting_bound")
        assert len(circuits) == 2
        assert all(isinstance(c, QuantumCircuit) for c in circuits)
        assert len(data_qubits) == 4

    def test_dfe_returns_one_circuit_per_stabilizer(self):
        graph = rx.generators.complete_graph(4)
        stabilizers = [[], [0, 1], [1, 2, 3, 0]]
        circuits, data_qubits = build_ghz_circuits(
            graph, num_qubits=4, method="dfe", stabilizers=stabilizers
        )
        # 1 z-basis + one circuit per sampled stabilizer
        assert len(circuits) == 4
        # Y-basis rotation (sdg) appears exactly on the support of each s.
        for qc, y_positions in zip(circuits[1:], stabilizers):
            sdg_qubits = {
                qc.find_bit(inst.qubits[0]).index
                for inst in qc.data
                if inst.operation.name == "sdg"
            }
            assert sdg_qubits == {data_qubits[i] for i in y_positions}

    def test_dfe_no_stabilizers_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="stabilizers required"):
            build_ghz_circuits(graph, num_qubits=4, method="dfe")

    def test_parity_oscillation_returns_correct_count(self):
        graph = rx.generators.complete_graph(4)
        phases = np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist()
        circuits, data_qubits = build_ghz_circuits(
            graph, num_qubits=4, method="parity_oscillation", phases=phases
        )
        # 1 z-basis + 10 oscillation circuits
        assert len(circuits) == 11
        assert len(data_qubits) == 4

    def test_compressed_sensing_returns_correct_count(self):
        graph = rx.generators.complete_graph(4)
        # CS uses a single-period grid; circuit shape matches parity_oscillation.
        phases = np.linspace(0, 2 * np.pi / 4, 6, endpoint=False).tolist()
        circuits, data_qubits = build_ghz_circuits(
            graph, num_qubits=4, method="compressed_sensing", phases=phases
        )
        # 1 z-basis + 6 oscillation circuits
        assert len(circuits) == 7
        assert len(data_qubits) == 4

    def test_compressed_sensing_no_phases_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="phases required"):
            build_ghz_circuits(graph, num_qubits=4, method="compressed_sensing")

    def test_layout_is_deterministic(self):
        # rustworkx neighbour order is not stable, so the selection must sort;
        # otherwise the same device yields different GHZ layouts run to run.
        graph = rx.generators.heavy_hex_graph(5)
        runs = [
            build_ghz_circuits(graph, num_qubits=10, method="two_setting_bound")[1]
            for _ in range(5)
        ]
        assert all(r == runs[0] for r in runs)

    def test_unknown_method_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="Unknown verification method"):
            build_ghz_circuits(graph, num_qubits=4, method="invalid")

    def test_parity_oscillation_no_phases_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="phases required"):
            build_ghz_circuits(graph, num_qubits=4, method="parity_oscillation")

    def test_num_qubits_exceeds_device_raises(self):
        graph = rx.generators.path_graph(3)
        with pytest.raises(ValueError, match="device only exposes"):
            build_ghz_circuits(graph, num_qubits=5, method="two_setting_bound")

    def test_unreachable_qubits_via_bfs_raises(self):
        # Two disjoint components of 3 nodes each: 0-1-2 and 3-4-5
        graph = rx.PyGraph()
        graph.add_nodes_from(range(6))
        graph.add_edges_from_no_data([(0, 1), (1, 2), (3, 4), (4, 5)])
        with pytest.raises(ValueError, match="connectivity graph may"):
            build_ghz_circuits(graph, num_qubits=6, method="two_setting_bound")


class TestEstimateFidelityTwoSettingBound:
    def test_perfect_ghz(self):
        n = 3
        z_counts = {"000": 500, "111": 500}
        x_counts = {
            "000": 250,
            "011": 250,
            "101": 250,
            "110": 250,
        }  # all even parity
        pop, coh, p_err, c_err = estimate_fidelity_two_setting_bound(z_counts, x_counts, n)
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0)

    def test_ghz_minus_treated_same_as_ghz_plus(self):
        # GHZ- = (|000> - |111>)/sqrt(2) — X-basis measurements give purely
        # odd-parity outcomes. Without abs() the coherence would come back as
        # -1.0 and the fidelity lower bound (pop + coh)/2 would collapse to 0
        # despite the state having perfect off-diagonal magnitude. (DFE proper
        # keeps the sign: fidelity against GHZ+ specifically, see
        # TestEstimateFidelityDfe.)
        n = 3
        z_counts = {"000": 500, "111": 500}
        x_counts = {
            "001": 250,
            "010": 250,
            "100": 250,
            "111": 250,
        }  # all odd parity
        pop, coh, _, _ = estimate_fidelity_two_setting_bound(z_counts, x_counts, n)
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0)

    def test_maximally_mixed(self):
        n = 2
        # Uniform distribution over all 4 bitstrings
        z_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        x_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        pop, coh, _, _ = estimate_fidelity_two_setting_bound(z_counts, x_counts, n)
        assert pop == pytest.approx(0.5)
        assert coh == pytest.approx(0.0)

    def test_empty_counts_returns_zero(self):
        pop, coh, p_err, c_err = estimate_fidelity_two_setting_bound({}, {}, 3)
        assert pop == 0.0
        assert coh == 0.0


class TestSampleGhzStabilizers:
    def test_even_weight_and_range(self):
        samples = sample_ghz_stabilizers(7, 200, seed=3)
        assert len(samples) == 200
        for s in samples:
            assert len(s) % 2 == 0
            assert all(0 <= i < 7 for i in s)
            assert s == sorted(set(s))

    def test_seed_reproducible(self):
        assert sample_ghz_stabilizers(6, 20, seed=5) == sample_ghz_stabilizers(6, 20, seed=5)
        assert sample_ghz_stabilizers(6, 20, seed=5) != sample_ghz_stabilizers(6, 20, seed=6)


class TestEstimateFidelityDfe:
    """Flammia-Liu direct fidelity estimation on statevector-exact counts."""

    def _exact_counts(self, qc, shots=100_000):
        from qiskit.quantum_info import Statevector

        bare = qc.remove_final_measurements(inplace=False)
        probs = Statevector(bare).probabilities_dict()
        return {b: int(round(p * shots)) for b, p in probs.items()}

    def _inject_after_prep(self, qc, gate, qubit):
        """Rebuild qc with `gate` on `qubit` right after the prep barrier."""
        new_qc = qc.copy_empty_like()
        injected = False
        for inst in qc.data:
            new_qc.append(inst)
            if not injected and inst.operation.name == "barrier":
                getattr(new_qc, gate)(qubit)
                injected = True
        assert injected
        return new_qc

    def _run_dfe(self, n, stabilizers, inject_gate=None):
        graph = rx.generators.complete_graph(n)
        circuits, data_qubits = build_ghz_circuits(
            graph, num_qubits=n, method="dfe", stabilizers=stabilizers
        )
        if inject_gate is not None:
            circuits = [self._inject_after_prep(qc, inject_gate, data_qubits[0]) for qc in circuits]
        counts = [self._exact_counts(qc) for qc in circuits]
        return estimate_fidelity_dfe(counts[0], counts[1:], stabilizers, n)

    def test_ideal_ghz_fidelity_one(self):
        # Include both sign classes: |s| % 4 == 0 (sign +1) and |s| % 4 == 2
        # (sign -1, where the raw parity of the ideal state is -1).
        n = 4
        stabilizers = [[], [0, 1], [1, 3], [0, 1, 2, 3]]
        pop, coh, p_err, c_err = self._run_dfe(n, stabilizers)
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0, abs=1e-6)
        assert (pop + coh) / 2 == pytest.approx(1.0, abs=1e-6)

    def test_ghz_minus_scores_zero(self):
        # A z on one data qubit after prep turns GHZ+ into GHZ-, which is
        # orthogonal to GHZ+, so an unbiased estimator must report fidelity 0:
        # population 1, coherence -1. The two-setting bound's |parity| would
        # report 1 here.
        n = 3
        stabilizers = [[], [0, 1], [0, 2], [1, 2]]
        pop, coh, _, _ = self._run_dfe(n, stabilizers, inject_gate="z")
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(-1.0, abs=1e-6)
        assert (pop + coh) / 2 == pytest.approx(0.0, abs=1e-6)

    def test_separable_plus_state_not_certified(self):
        # For |+>^n every off-diagonal stabilizer with s != 0 has expectation
        # zero, so the sampled coherence stays near zero and the fidelity
        # cannot cross 0.5, unlike the raw X-parity which is 1.
        n = 6
        stabilizers = [[0, 1], [2, 3], [0, 3], [1, 2, 4, 5]]
        z_counts = {
            "".join(bits): 64 for bits in itertools.product("01", repeat=n)
        }  # uniform Z distribution of |+>^n
        # Each Y on |+> gives ±1 with equal probability: uniform parity.
        stab_counts = [{"0" * n: 500, "0" * (n - 1) + "1": 500} for _ in stabilizers]
        pop, coh, _, _ = estimate_fidelity_dfe(z_counts, stab_counts, stabilizers, n)
        assert pop == pytest.approx(2 ** (1 - n))
        assert coh == pytest.approx(0.0, abs=1e-9)
        assert (pop + coh) / 2 < 0.5

    def test_empty_counts_returns_zero(self):
        pop, coh, p_err, c_err = estimate_fidelity_dfe({}, [], [], 3)
        assert pop == 0.0
        assert coh == 0.0


class TestEstimateFidelityOscillation:
    def test_perfect_oscillation(self):
        n = 4
        phases = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        # Perfect oscillation: cos(4*phi)
        osc_counts_list = []
        for phi in phases:
            parity = np.cos(n * phi)
            # Convert parity to counts: even_frac = (1 + parity) / 2
            total = 1000
            even = int(total * (1 + parity) / 2)
            odd = total - even
            # Real GHZ circuits return n-bit measurement outcomes — use any
            # even-parity / odd-parity 4-bit labels here. The estimator only
            # counts "1"s in each bitstring, so labels are length-agnostic,
            # but matching the real input shape avoids confusing future readers.
            osc_counts_list.append({"0000": even, "0001": odd})

        z_counts = {"0000": 500, "1111": 500}
        pop, coh, _, _ = estimate_fidelity_oscillation(
            z_counts, osc_counts_list, phases.tolist(), n
        )
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0, abs=0.05)

    def test_empty_z_counts(self):
        pop, coh, _, _ = estimate_fidelity_oscillation({}, [], [], 3)
        assert pop == 0.0
        assert coh == 0.0


class TestEstimateFidelityCompressedSensing:
    def _random_phases(self, count: int, seed: int = 7) -> list[float]:
        """Random phases on [0, 2π), as drawn by the CS phase grid."""
        return np.random.default_rng(seed).uniform(0, 2 * np.pi, count).tolist()

    def _build_osc_counts(self, n: int, phases, shots: int, parity_fn):
        """Helper: synthesize parity counts from P(φ) = parity_fn(φ)."""
        osc_counts_list = []
        for phi in phases:
            parity = parity_fn(phi)
            even = int(round(shots * (1 + parity) / 2))
            odd = shots - even
            osc_counts_list.append({"0" * n: even, "0" * (n - 1) + "1": odd})
        return osc_counts_list

    def test_perfect_ghz_amplitude_one(self):
        n = 4
        # Sub-Nyquist sampling: M = 11 random phases vs 2n+1 = 9 unknowns
        # spread over frequencies up to n.
        phases = self._random_phases(11)
        osc_counts_list = self._build_osc_counts(
            n, phases, shots=2000, parity_fn=lambda phi: np.cos(n * phi)
        )

        z_counts = {"0" * n: 1000, "1" * n: 1000}
        pop, coh, _p_err, c_err, freq, phase, phase_err = estimate_fidelity_compressed_sensing(
            z_counts, osc_counts_list, phases, n
        )
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0, abs=0.02)
        assert freq == n
        assert c_err >= 0.0
        # An ideal GHZ signal cos(nφ) carries no phase offset.
        assert phase == pytest.approx(0.0, abs=0.05)
        assert phase_err is not None and phase_err >= 0.0

    def test_recovers_amplitude_below_one(self):
        n = 5
        phases = self._random_phases(13)
        target_amplitude = 0.6
        osc_counts_list = self._build_osc_counts(
            n, phases, shots=5000, parity_fn=lambda phi: target_amplitude * np.cos(n * phi + 0.7)
        )
        z_counts = {"0" * n: 800, "1" * n: 200}
        pop, coh, _p_err, _c_err, freq, phase, _phase_err = estimate_fidelity_compressed_sensing(
            z_counts, osc_counts_list, phases, n
        )
        # Population reflects whatever Z-basis stats the user provided.
        assert pop == pytest.approx(1.0)
        # CS estimates magnitude regardless of phase offset.
        assert coh == pytest.approx(target_amplitude, abs=0.03)
        assert freq == n
        # The signal cos(nφ + 0.7) carries phase offset 0.7.
        assert phase == pytest.approx(0.7, abs=0.05)

    def test_broken_ghz_recovers_actual_size(self):
        # Intended 12-qubit GHZ but only a 10-qubit entangled core, with the
        # remaining qubits in |+>. The parity signal is cos(10φ)·cos²(φ),
        # which has spectral peaks at frequencies 8, 10 (dominant), and 12.
        # The dominant recovered frequency exposes the actual GHZ size, and
        # the coherence at frequency 12 drops to 1/4.
        n, k = 12, 10
        phases = self._random_phases(30)
        osc_counts_list = self._build_osc_counts(
            n, phases, shots=100_000, parity_fn=lambda phi: np.cos(k * phi) * np.cos(phi) ** 2
        )
        z_counts = {"0" * n: 500, "1" * n: 500}
        _pop, coh, _p_err, _c_err, freq, _phase, _phase_err = estimate_fidelity_compressed_sensing(
            z_counts, osc_counts_list, phases, n
        )
        assert freq == k
        assert coh == pytest.approx(0.25, abs=0.03)

    def test_zero_parity_signal_gives_zero_coherence(self):
        n = 3
        phases = self._random_phases(8)
        # P(φ) = 0 for every phase ⇒ empty spectrum.
        osc_counts_list = [{"000": 500, "001": 500} for _ in phases]
        z_counts = {"000": 250, "001": 250, "010": 250, "011": 250}
        pop, coh, _p_err, _c_err, freq, phase, phase_err = estimate_fidelity_compressed_sensing(
            z_counts, osc_counts_list, phases, n
        )
        assert coh == pytest.approx(0.0, abs=0.02)
        assert pop == pytest.approx(0.25)
        assert freq == 0
        # No signal means the phase is undefined, not atan2 of noise.
        assert phase is None
        assert phase_err is None

    def test_empty_z_counts(self):
        pop, coh, _, _, freq, phase, phase_err = estimate_fidelity_compressed_sensing({}, [], [], 3)
        assert pop == 0.0
        assert coh == 0.0
        assert freq == 0
        assert phase is None
        assert phase_err is None

    @pytest.mark.parametrize("delta", [0.9, -1.3])
    def test_phase_offset_sign_convention_from_circuits(self, delta):
        # Inject rz(delta) on one data qubit right after GHZ preparation,
        # turning the state into (|000> + e^{i·delta}|111>)/√2. The recovered
        # phase offset must come back as +delta — this pins the sign convention
        # against the actual measurement circuits rather than a synthetic
        # signal that would bake the convention into the test itself.
        from qiskit.quantum_info import Statevector

        n = 3
        graph = rx.generators.path_graph(n)
        phases = self._random_phases(9)
        circuits, data_qubits = build_ghz_circuits(
            graph, num_qubits=n, method="compressed_sensing", phases=phases
        )

        def exact_counts(qc, shots=200_000):
            bare = qc.remove_final_measurements(inplace=False)
            probs = Statevector(bare).probabilities_dict()
            return {b: int(round(p * shots)) for b, p in probs.items()}

        def inject_after_prep(qc):
            injected = qc.copy_empty_like()
            done = False
            for inst in qc.data:
                if not done and inst.operation.name == "rz":
                    injected.rz(delta, data_qubits[0])
                    done = True
                injected.append(inst)
            return injected

        z_counts = exact_counts(circuits[0])
        osc_counts = [exact_counts(inject_after_prep(qc)) for qc in circuits[1:]]

        *_, phase, phase_err = estimate_fidelity_compressed_sensing(z_counts, osc_counts, phases, n)
        assert phase == pytest.approx(delta, abs=0.05)
        assert phase_err is not None and phase_err >= 0.0


class TestPhaseGrid:
    def _benchmark(self, **params) -> GHZBenchmark:
        return GHZBenchmark(args=Namespace(), params=Namespace(**params))

    def test_parity_oscillation_uniform_grid(self):
        bench = self._benchmark(num_phases=None)
        grid = bench._phase_grid("parity_oscillation", n=8)
        assert grid == pytest.approx(np.linspace(0, 2 * np.pi, 20, endpoint=False).tolist())

    def test_compressed_sensing_default_count_is_log_scaled(self):
        bench = self._benchmark(num_phases=None, seed=1)
        grid = bench._phase_grid("compressed_sensing", n=8)
        # M = max(6, ceil(5 ln 8)) = 11 random phases on [0, 2π)
        assert len(grid) == 11
        assert all(0 <= phi < 2 * np.pi for phi in grid)

    def test_compressed_sensing_seed_reproducible(self):
        grid_a = self._benchmark(num_phases=None, seed=42)._phase_grid("compressed_sensing", n=6)
        grid_b = self._benchmark(num_phases=None, seed=42)._phase_grid("compressed_sensing", n=6)
        grid_c = self._benchmark(num_phases=None, seed=43)._phase_grid("compressed_sensing", n=6)
        assert grid_a == grid_b
        assert grid_a != grid_c

    def test_explicit_num_phases_respected(self):
        bench = self._benchmark(num_phases=25, seed=0)
        assert len(bench._phase_grid("compressed_sensing", n=8)) == 25
        assert len(bench._phase_grid("parity_oscillation", n=8)) == 25

    def test_dfe_has_no_phases(self):
        bench = self._benchmark(num_phases=None)
        assert bench._phase_grid("dfe", n=8) == []
        assert bench._phase_grid("two_setting_bound", n=8) == []


class TestGHZResult:
    def test_compute_score(self):
        result = GHZResult(
            population=BenchmarkScore(value=0.9, uncertainty=0.01),
            coherence=BenchmarkScore(value=0.8, uncertainty=0.02),
            fidelity=BenchmarkScore(value=0.85, uncertainty=0.01),
        )
        assert result.compute_score() == result.fidelity

    def test_values_dict(self):
        result = GHZResult(
            population=BenchmarkScore(value=0.9, uncertainty=0.01),
            coherence=BenchmarkScore(value=0.8, uncertainty=0.02),
            fidelity=BenchmarkScore(value=0.85, uncertainty=0.01),
        )
        vals = result.values
        assert "population" in vals
        assert "coherence" in vals
        assert "fidelity" in vals


class TestBisectionSizes:
    def test_halving_grid(self):
        assert bisection_sizes(156, 5) == [156, 78, 39, 19, 9, 5]

    def test_device_at_lower_bound(self):
        assert bisection_sizes(5, 5) == [5]

    def test_device_below_lower_bound_raises(self):
        with pytest.raises(ValueError, match="below the search lower bound"):
            bisection_sizes(4, 5)

    def test_all_sizes_at_least_min(self):
        for total in (7, 31, 100, 1000):
            sizes = bisection_sizes(total, 5)
            assert sizes[0] == total
            assert all(s >= 5 for s in sizes)
            assert sizes == sorted(sizes, reverse=True)


class TestSizeSearchPoll:
    def _two_setting_counts(self, n: int, fidelity_high: bool) -> list[dict[str, int]]:
        """Synthesize Z/X counts: ideal GHZ when high, maximally mixed when low."""
        if fidelity_high:
            z = {"0" * n: 500, "1" * n: 500}
            # X-basis: all even-parity outcomes
            x = {"0" * n: 500, "0" * (n - 2) + "11": 500}
        else:
            # Population 0 (no all-zero/all-one outcomes) and coherence 0
            # (even and odd X-basis parity equally likely).
            z = {"0" * (n - 1) + "1": 1000}
            x = {"0" * n: 500, "0" * (n - 1) + "1": 500}
        return [z, x]

    def _make_benchmark(self):
        from argparse import Namespace

        return GHZBenchmark(args=Namespace(), params=Namespace())

    def test_largest_passing_size_found(self):
        from metriq_gym.benchmarks.ghz import GHZData

        # Device of 8: sizes [8, 4]; 8 fails, 4 passes.
        counts = self._two_setting_counts(8, False) + self._two_setting_counts(4, True)
        job_data = GHZData(
            provider_job_ids=["x"],
            num_qubits=8,
            method="two_setting_bound",
            search_sizes=[8, 4],
            search_circuit_counts=[2, 2],
            search_phases=[[], []],
            fidelity_threshold=0.5,
        )
        result = self._make_benchmark()._poll_size_search(job_data, counts, "two_setting_bound")
        assert result.largest_passing_size is not None
        assert result.largest_passing_size.value == 4
        assert result.device_fraction is not None
        assert result.device_fraction.value == pytest.approx(0.5)
        assert result.search_sizes == [8, 4]
        assert result.search_fidelities[0] < 0.5 < result.search_fidelities[1]
        # Headline fidelity describes the passing size.
        assert result.fidelity.value == pytest.approx(result.search_fidelities[1])
        assert result.compute_score().value == 4

    def test_no_size_passes(self):
        from metriq_gym.benchmarks.ghz import GHZData

        counts = self._two_setting_counts(8, False) + self._two_setting_counts(4, False)
        job_data = GHZData(
            provider_job_ids=["x"],
            num_qubits=8,
            method="two_setting_bound",
            search_sizes=[8, 4],
            search_circuit_counts=[2, 2],
            search_phases=[[], []],
            fidelity_threshold=0.5,
        )
        result = self._make_benchmark()._poll_size_search(job_data, counts, "two_setting_bound")
        assert result.largest_passing_size.value == 0
        assert result.device_fraction.value == 0
        assert result.compute_score().value == 0

    def test_full_device_passes(self):
        from metriq_gym.benchmarks.ghz import GHZData

        counts = self._two_setting_counts(8, True) + self._two_setting_counts(4, True)
        job_data = GHZData(
            provider_job_ids=["x"],
            num_qubits=8,
            method="two_setting_bound",
            search_sizes=[8, 4],
            search_circuit_counts=[2, 2],
            search_phases=[[], []],
            fidelity_threshold=0.5,
        )
        result = self._make_benchmark()._poll_size_search(job_data, counts, "two_setting_bound")
        assert result.largest_passing_size.value == 8
        assert result.device_fraction.value == pytest.approx(1.0)


class TestSeparableStateNotCertified:
    """A product state has full global X parity but no entanglement.

    |+>^N measures deterministically to all-zero in the X basis, so the raw
    parity is 1. Reporting that as the GHZ coherence gave fidelity 0.504 for
    N=8, which cleared the 0.5 size-search threshold on a state containing no
    entanglement at all.
    """

    def _plus_state_counts(self, n: int, shots: int):
        z = {"".join(bits): shots // 2**n for bits in itertools.product("01", repeat=n)}
        x = {"0" * n: shots}
        return z, x

    def test_plus_state_reports_true_fidelity(self):
        n = 8
        z, x = self._plus_state_counts(n, 2**n * 4)
        pop, coh, _, _ = estimate_fidelity_two_setting_bound(z, x, n)
        fidelity = (pop + coh) / 2
        # <GHZ|+^N>^2 = 2^(1-N)
        assert fidelity == pytest.approx(2 ** (1 - n), abs=1e-9)
        assert fidelity < 0.5

    def test_ideal_ghz_still_certified(self):
        n = 8
        shots = 1000
        z = {"0" * n: shots // 2, "1" * n: shots // 2}
        x = {"0" * n: shots}
        pop, coh, _, _ = estimate_fidelity_two_setting_bound(z, x, n)
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0)

    def test_sampling_adequacy_gate(self):
        # The correction is estimated from observed Z frequencies, so it stops
        # being a valid bound once 2^n outruns the shots.
        assert two_setting_sampling_is_adequate(8, 1000) is True
        assert two_setting_sampling_is_adequate(50, 1000) is False


class TestPreparationDepth:
    def test_all_to_all_is_logarithmic(self):
        # A fixed-root star runs every CNOT off qubit 0 and serialises.
        n = 16
        graph = rx.generators.complete_graph(n)
        circuits, _ = build_ghz_circuits(graph, num_qubits=n, method="two_setting_bound")
        assert circuits[0].depth() < n // 2

    def test_path_still_builds(self):
        graph = rx.generators.path_graph(8)
        circuits, data_qubits = build_ghz_circuits(graph, num_qubits=8, method="two_setting_bound")
        assert len(data_qubits) == 8


class TestPhaseGridAliasing:
    def _bench(self, **kw):
        return GHZBenchmark(args=Namespace(), params=Namespace(**kw))

    def test_default_scales_with_n(self):
        grid = self._bench(num_phases=None)._phase_grid("parity_oscillation", 100)
        assert len(grid) > 2 * 100

    def test_aliasing_value_rejected(self):
        # At 20 uniform angles cos(100 phi) and cos(80 phi) are identical.
        with pytest.raises(ValueError, match="aliasing"):
            self._bench(num_phases=20)._phase_grid("parity_oscillation", 100)

    def test_small_state_keeps_previous_default(self):
        assert len(self._bench(num_phases=None)._phase_grid("parity_oscillation", 4)) == 20


class TestSizeSearchDefaults:
    def _bench(self, **kw):
        return GHZBenchmark(args=Namespace(), params=Namespace(**kw))

    def test_size_search_defaults_to_compressed_sensing(self):
        assert self._bench(size_search=True, method=None)._resolved_method() == "compressed_sensing"

    def test_fixed_size_defaults_to_dfe(self):
        assert self._bench(size_search=False, method=None)._resolved_method() == "dfe"

    def test_explicit_method_respected(self):
        assert self._bench(size_search=True, method="dfe")._resolved_method() == "dfe"
