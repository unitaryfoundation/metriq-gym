from qedc.bernstein_vazirani.bv_benchmark import run


# A basic test to ensure that circuits are being made
def testGettingCircs():
    circuits, metrics = run(
        min_qubits=2,
        max_qubits=6,
        skip_qubits=1,
        max_circuits=3,
        num_shots=1,
        method=1,
        get_circuits=True,
    )

    print(f"\n\n{circuits}\n\n")

    assert len(circuits) > 0
