"""Abstract base class for quantum error mitigation techniques."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@dataclass
class MitigationMetadata:
    """Opaque metadata produced by transform_circuits, consumed by postprocess_counts.

    Stores technique-specific data needed to reverse or interpret the circuit
    transformations during post-processing. Must be JSON-serializable for
    persistence in MetriqGymJob.data.
    """

    technique_name: str
    data: dict[str, Any] = field(default_factory=dict)


class MitigationTechnique(ABC):
    """Abstract base class for quantum error mitigation techniques.

    Each technique implements two hooks:
    - transform_circuits: Modify or expand circuits before submission.
    - postprocess_counts: Process measurement counts after retrieval.

    Techniques are composed via MitigationPipeline, which applies transforms
    in order and post-processing in reverse order.
    """

    name: str

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def transform_circuits(
        self,
        circuits: list["QuantumCircuit"],
        shots: int,
    ) -> tuple[list["QuantumCircuit"], MitigationMetadata]:
        """Transform circuits before submission to the quantum device.

        May return more circuits than input (e.g., ZNE noise-scaled copies),
        modified circuits (e.g., DD pulse insertion), or unchanged circuits
        (e.g., REM which only post-processes).

        Args:
            circuits: Original benchmark circuits.
            shots: Number of shots per circuit.

        Returns:
            Tuple of (transformed_circuits, metadata_for_postprocessing).
        """
        ...

    @abstractmethod
    def postprocess_counts(
        self,
        counts: list[dict[str, int]],
        metadata: MitigationMetadata,
    ) -> list[dict[str, int]]:
        """Post-process measurement counts after retrieval.

        Must return counts in the same shape as the ORIGINAL circuits
        (before transform_circuits expanded them).

        Args:
            counts: Measurement counts from the (possibly transformed) circuits.
            metadata: Metadata produced by transform_circuits for this technique.

        Returns:
            Post-processed counts matching the original circuit count.
        """
        ...
