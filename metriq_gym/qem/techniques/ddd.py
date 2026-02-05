"""Digital Dynamical Decoupling (DDD) technique using mitiq.

DDD inserts refocusing gate sequences into idle periods (slack windows) in circuits
to suppress dephasing and other coherent errors.
"""

from typing import Any, TYPE_CHECKING

from metriq_gym.qem.technique import MitigationMetadata, MitigationTechnique
from metriq_gym.qem.registry import register_technique

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@register_technique("ddd")
class DDDTechnique(MitigationTechnique):
    """Digital Dynamical Decoupling via idle-period gate insertion.

    DDD inserts sequences of gates (e.g., XYXY, XX, YY) into slack windows
    where qubits would otherwise be idle. These sequences refocus dephasing
    errors, improving circuit fidelity without post-processing.

    Config options:
        rule: str - The DDD rule/sequence to use: "xyxy" (default), "xx", or "yy".
        spacing: int - Spacing between decoupling gates within a sequence.
            Default -1 means mitiq will use maximal spacing to fill the slack window.
    """

    name = "ddd"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.rule_name: str = config.get("rule", "xyxy")
        self.spacing: int = config.get("spacing", -1)

        valid_rules = ("xyxy", "xx", "yy")
        if self.rule_name not in valid_rules:
            raise ValueError(
                f"Unknown DDD rule '{self.rule_name}'. Must be one of {valid_rules}"
            )

    def _get_rule(self):
        """Get the mitiq DDD rule function."""
        from mitiq.ddd import rules

        rule_map = {
            "xyxy": rules.xyxy,
            "xx": rules.xx,
            "yy": rules.yy,
        }
        base_rule = rule_map[self.rule_name]

        # If custom spacing is specified, wrap the rule
        if self.spacing != -1:
            from functools import partial

            return partial(base_rule, spacing=self.spacing)
        return base_rule

    def transform_circuits(
        self,
        circuits: list["QuantumCircuit"],
        shots: int,
    ) -> tuple[list["QuantumCircuit"], MitigationMetadata]:
        """Insert DDD sequences into idle periods of each circuit.

        For N original circuits, produces N transformed circuits (1:1 mapping).
        No circuit expansion occurs — DDD only modifies existing circuits.
        """
        from mitiq.ddd import insert_ddd_sequences

        rule = self._get_rule()
        transformed_circuits: list["QuantumCircuit"] = []

        for circuit in circuits:
            # mitiq's insert_ddd_sequences works directly on qiskit circuits
            ddd_circuit = insert_ddd_sequences(circuit, rule)
            transformed_circuits.append(ddd_circuit)

        metadata = MitigationMetadata(
            technique_name=self.name,
            data={
                "rule": self.rule_name,
                "spacing": self.spacing,
                "original_circuit_count": len(circuits),
            },
        )
        return transformed_circuits, metadata

    def postprocess_counts(
        self,
        counts: list[dict[str, int]],
        metadata: MitigationMetadata,
    ) -> list[dict[str, int]]:
        """DDD requires no post-processing — return counts unchanged.

        All mitigation happens at the circuit level via gate insertion.
        """
        return counts
