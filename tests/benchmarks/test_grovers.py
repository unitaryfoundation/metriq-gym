from metriq_gym.benchmarks.grovers.grovers import create_circuits, calc_fidelities, GroversData


def test_create_circuits():
    """
    A basic test to ensure that there are no errors in creating the circuits.
    Test passed; all variables are populated.
    Note that there are many deprecation warnings that should be addressed.
    """
    grovers_circuits, marked_items, all_num_qubits = create_circuits(
        min_qubits=2, max_qubits=6, skip_qubits=1, max_circuits=3, use_mcx_shim=False
    )

    print(f"\n\n{grovers_circuits} \n\n {marked_items} \n\n {all_num_qubits}")

    assert len(grovers_circuits) == len(all_num_qubits) * 3
    assert len(marked_items) == len(all_num_qubits)
    for i in range(len(marked_items)):
        assert len(marked_items[i]) == 3
    assert len(all_num_qubits) == 5


def test_calc_fidelities():
    """
    A basic test to ensure that there are no errors in computing fidelities.
    Test passed; the fidelities list is populated.
    Note that the numbers have no meaning, they are made-up results.
    """

    job_data = GroversData(
        provider_job_ids=["test_job_id"],
        shots=100,
        min_qubits=2,
        max_qubits=3,
        skip_qubits=1,
        max_circuits=3,
        marked_items=[[1, 2, 3], [1, 2, 4]],
        all_num_qubits=[2, 3],
        use_mcx_shim=False,
    )

    counts = [
        {"00": 80, "01": 20, "10": 0, "11": 0},
        {"00": 80, "01": 20, "10": 0, "11": 0},
        {"00": 80, "01": 20, "10": 0, "11": 0},
        {"000": 6, "001": 6, "010": 6, "011": 6, "100": 58, "101": 6, "110": 6, "111": 6},
        {"000": 6, "001": 6, "010": 6, "011": 6, "100": 58, "101": 6, "110": 6, "111": 6},
        {"000": 6, "001": 6, "010": 6, "011": 6, "100": 58, "101": 6, "110": 6, "111": 6},
    ]

    fidelities = calc_fidelities(job_data, counts)

    print(f"\n\n{fidelities}")

    assert len(fidelities) > 0
