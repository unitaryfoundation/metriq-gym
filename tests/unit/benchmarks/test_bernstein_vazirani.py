from qedc.bernstein_vazirani.bv_benchmark import run


def test_run_bv_circuits():
    """A basic test to ensure that circuits are being made."""
    circuits, _ = run(
        min_qubits=2,
        max_qubits=6,
        skip_qubits=1,
        max_circuits=3,
        num_shots=1,
        method=1,
        get_circuits=True,
    )

    assert len(circuits) > 0
