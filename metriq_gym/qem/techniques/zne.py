"""Zero Noise Extrapolation (ZNE) technique using mitiq.

ZNE works by running circuits at multiple noise levels (via gate folding)
and extrapolating the results to the zero-noise limit.
"""

from typing import Any, TYPE_CHECKING

from metriq_gym.qem.technique import MitigationMetadata, MitigationTechnique
from metriq_gym.qem.registry import register_technique

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@register_technique("zne")
class ZNETechnique(MitigationTechnique):
    """Zero Noise Extrapolation via noise-scaled circuit copies.

    Config options:
        scale_factors: list[float] - Noise scale factors (default: [1.0, 2.0, 3.0]).
            Must include 1.0 (the unscaled case) and have at least 2 entries.
        factory: str - Extrapolation method: "linear", "richardson", or "poly"
            (default: "linear").
        fold_method: str - Circuit folding method: "global", "gates_at_random",
            or "gates_from_left" (default: "global").
        degree: int - Polynomial degree for "poly" factory (default: 2).
    """

    name = "zne"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.scale_factors: list[float] = [
            float(s) for s in config.get("scale_factors", [1.0, 2.0, 3.0])
        ]
        self.factory_type: str = config.get("factory", "linear")
        self.fold_method: str = config.get("fold_method", "global")
        self.degree: int = config.get("degree", 2)

        if len(self.scale_factors) < 2:
            raise ValueError("ZNE requires at least 2 scale factors")
        if 1.0 not in self.scale_factors:
            raise ValueError("ZNE scale_factors must include 1.0 (the unscaled case)")
        valid_factories = ("linear", "richardson", "poly")
        if self.factory_type not in valid_factories:
            raise ValueError(
                f"Unknown ZNE factory '{self.factory_type}'. Must be one of {valid_factories}"
            )
        valid_folds = ("global", "gates_at_random", "fold_all")
        if self.fold_method not in valid_folds:
            raise ValueError(
                f"Unknown fold method '{self.fold_method}'. Must be one of {valid_folds}"
            )

    def _get_fold_function(self):
        from mitiq.zne.scaling import (
            fold_global,
            fold_gates_at_random,
            fold_all,
        )

        return {
            "global": fold_global,
            "gates_at_random": fold_gates_at_random,
            "fold_all": fold_all,
        }[self.fold_method]

    def _create_factory(self):
        from mitiq.zne.inference import (
            LinearFactory,
            RichardsonFactory,
            PolyFactory,
        )

        factories = {
            "linear": lambda: LinearFactory(self.scale_factors),
            "richardson": lambda: RichardsonFactory(self.scale_factors),
            "poly": lambda: PolyFactory(self.scale_factors, order=self.degree),
        }
        return factories[self.factory_type]()

    @staticmethod
    def _strip_measurements(
        circuit: "QuantumCircuit",
    ) -> tuple["QuantumCircuit", list[tuple[int, int]]]:
        """Remove final measurements from a circuit for folding.

        Mitiq's folding functions require circuits without measurements.

        Returns:
            Tuple of (circuit_without_measurements, measurement_map).
            measurement_map is a list of (qubit_index, clbit_index) pairs,
            or an empty list if no measurements were found.
        """
        has_meas = any(
            inst.operation.name == "measure" for inst in circuit.data
        )
        if not has_meas:
            return circuit, []

        # Save the qubit→clbit measurement mapping
        measurements: list[tuple[int, int]] = []
        for inst in circuit.data:
            if inst.operation.name == "measure":
                q_idx = circuit.find_bit(inst.qubits[0]).index
                c_idx = circuit.find_bit(inst.clbits[0]).index
                measurements.append((q_idx, c_idx))

        # Use Qiskit's built-in to cleanly remove measurements and unused clbits
        stripped = circuit.remove_final_measurements(inplace=False)
        return stripped, measurements

    @staticmethod
    def _restore_measurements(
        circuit: "QuantumCircuit",
        measurements: list[tuple[int, int]],
    ) -> "QuantumCircuit":
        """Restore the original measurements to a folded circuit."""
        if not measurements:
            return circuit

        from qiskit.circuit import ClassicalRegister

        num_clbits = max(c_idx for _, c_idx in measurements) + 1
        cr = ClassicalRegister(num_clbits)
        circuit.add_register(cr)
        for q_idx, c_idx in measurements:
            circuit.measure(circuit.qubits[q_idx], cr[c_idx])
        return circuit

    def transform_circuits(
        self,
        circuits: list["QuantumCircuit"],
        shots: int,
    ) -> tuple[list["QuantumCircuit"], MitigationMetadata]:
        """Create noise-scaled copies of each circuit.

        For N original circuits and K scale factors, produces N*K circuits
        in interleaved order: [c0_s0, c0_s1, ..., c0_sK, c1_s0, ...].
        """
        from mitiq.interface.mitiq_qiskit.conversions import from_qiskit, to_qiskit

        fold_fn = self._get_fold_function()
        scaled_circuits: list["QuantumCircuit"] = []

        for circuit in circuits:
            stripped, measurements = self._strip_measurements(circuit)
            for scale in self.scale_factors:
                if scale == 1.0:
                    folded = stripped.copy()
                else:
                    # Convert to cirq for mitiq, fold, convert back
                    cirq_circuit = from_qiskit(stripped)
                    folded_cirq = fold_fn(cirq_circuit, scale)
                    folded = to_qiskit(folded_cirq)

                if measurements:
                    folded = self._restore_measurements(folded, measurements)
                scaled_circuits.append(folded)

        metadata = MitigationMetadata(
            technique_name=self.name,
            data={
                "scale_factors": self.scale_factors,
                "factory_type": self.factory_type,
                "degree": self.degree,
                "fold_method": self.fold_method,
                "original_circuit_count": len(circuits),
            },
        )
        return scaled_circuits, metadata

    def postprocess_counts(
        self,
        counts: list[dict[str, int]],
        metadata: MitigationMetadata,
    ) -> list[dict[str, int]]:
        """Extrapolate zero-noise results from noise-scaled counts.

        For each original circuit, extrapolates per-bitstring probabilities
        to the zero-noise limit using the configured factory, then converts
        back to integer counts.
        """
        scale_factors = metadata.data["scale_factors"]
        original_count = metadata.data["original_circuit_count"]
        num_scales = len(scale_factors)

        mitigated_counts: list[dict[str, int]] = []
        for i in range(original_count):
            circuit_counts = counts[i * num_scales : (i + 1) * num_scales]
            mitigated_counts.append(
                self._extrapolate_single(circuit_counts, scale_factors)
            )

        return mitigated_counts

    def _extrapolate_single(
        self,
        circuit_counts: list[dict[str, int]],
        scale_factors: list[float],
    ) -> dict[str, int]:
        """Extrapolate counts for a single original circuit across scale factors."""
        # Collect all bitstrings across scale factors
        all_bitstrings: set[str] = set()
        totals: list[int] = []
        for c in circuit_counts:
            all_bitstrings.update(c.keys())
            totals.append(sum(c.values()))

        # Use unscaled shot count as reference
        unscaled_idx = scale_factors.index(1.0)
        total_shots = totals[unscaled_idx]

        # Extrapolate probability of each bitstring to zero noise
        extrapolated_probs: dict[str, float] = {}
        for bitstring in all_bitstrings:
            probs_at_scales = [
                c.get(bitstring, 0) / t if t > 0 else 0.0
                for c, t in zip(circuit_counts, totals)
            ]
            factory = self._create_factory()
            for scale, prob in zip(scale_factors, probs_at_scales):
                factory.push({"scale_factor": scale}, prob)
            extrapolated_probs[bitstring] = factory.reduce()

        # Clamp negatives and renormalize
        for bs in extrapolated_probs:
            extrapolated_probs[bs] = max(0.0, extrapolated_probs[bs])

        total_prob = sum(extrapolated_probs.values())
        if total_prob > 0:
            mitigated = {
                bs: max(0, round(total_shots * prob / total_prob))
                for bs, prob in extrapolated_probs.items()
                if round(total_shots * prob / total_prob) > 0
            }
        else:
            # Fallback to unscaled counts if extrapolation fails entirely
            mitigated = circuit_counts[unscaled_idx]

        return mitigated
