from dataclasses import dataclass, field


@dataclass
class _FakeProfile:
    basis_gates: list[str] | None = None


@dataclass
class _FakeDevice:
    id: str = "test-simulator"
    profile: _FakeProfile = field(default_factory=_FakeProfile)
    num_qubits: int | None = None
