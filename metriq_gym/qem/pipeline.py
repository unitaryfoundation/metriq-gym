"""Composable pipeline for chaining multiple QEM techniques."""

from typing import TYPE_CHECKING

from metriq_gym.qem.technique import MitigationMetadata, MitigationTechnique

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


class MitigationPipeline:
    """Composes multiple MitigationTechniques in sequence.

    Circuit transforms are applied in order (first technique first).
    Post-processing is applied in reverse order (last technique first),
    so the overall pipeline is: transform1 -> transform2 -> execute -> postprocess2 -> postprocess1.
    """

    def __init__(self, techniques: list[MitigationTechnique]):
        self.techniques = techniques

    def transform_circuits(
        self,
        circuits: list["QuantumCircuit"],
        shots: int,
    ) -> tuple[list["QuantumCircuit"], list[MitigationMetadata]]:
        """Apply all techniques' circuit transforms in order."""
        all_metadata: list[MitigationMetadata] = []
        current_circuits = circuits
        for technique in self.techniques:
            current_circuits, metadata = technique.transform_circuits(current_circuits, shots)
            all_metadata.append(metadata)
        return current_circuits, all_metadata

    def postprocess_counts(
        self,
        counts: list[dict[str, int]],
        all_metadata: list[MitigationMetadata],
    ) -> list[dict[str, int]]:
        """Apply all techniques' post-processing in reverse order."""
        current_counts = counts
        for technique, metadata in zip(
            reversed(self.techniques), reversed(all_metadata)
        ):
            current_counts = technique.postprocess_counts(current_counts, metadata)
        return current_counts

    @property
    def is_empty(self) -> bool:
        return len(self.techniques) == 0
