from dataclasses import dataclass, field

from metriq_gym.constants import JobType
from metriq_gym.resource_estimation import estimate_resources, quantinuum_hqc_formula
from metriq_gym.schema_validator import validate_and_create_model


@dataclass
class _FakeProfile:
    basis_gates: list[str] | None = None


@dataclass
class _FakeDevice:
    id: str = "test-simulator"
    profile: _FakeProfile = field(default_factory=_FakeProfile)
    num_qubits: int | None = None


def test_estimate_wit_resource_counts():
    params = validate_and_create_model({"benchmark_name": "WIT", "num_qubits": 6, "shots": 32})
    device = _FakeDevice()

    estimate = estimate_resources(JobType.WIT, params, device)

    assert estimate.circuit_count == 1
    assert estimate.total_shots == 32
    assert estimate.total_gate_counts.two_qubit > 0
    assert estimate.total_gate_counts.one_qubit > 0
    assert estimate.total_gate_counts.measurements == 1
    assert estimate.hqc_total is None

    estimate_with_hqc = estimate_resources(JobType.WIT, params, device, quantinuum_hqc_formula)

    counts = estimate.total_gate_counts
    expected_hqc = quantinuum_hqc_formula(counts, 32)

    assert estimate_with_hqc.hqc_total is not None
    assert abs(estimate_with_hqc.hqc_total - expected_hqc) < 1e-6
    assert estimate_with_hqc.per_circuit[0].hqc == estimate_with_hqc.hqc_total
