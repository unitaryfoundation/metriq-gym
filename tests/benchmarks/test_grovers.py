from metriq_gym.benchmarks.grovers.grovers import create_circuits, calc_fidelities, GroversData


def test_create_circuits():
    """
    A basic test to ensure that there are no errors in creating the circuits.
    Test passed; all variables are populated.
    Note that there are many deprecation warnings that should be addressed.
    """
    grovers_circuits, marked_items, all_num_qubits = create_circuits()

    # print(marked_items, "\n", "\n", all_num_qubits)

    assert len(grovers_circuits) > 0
    assert len(marked_items) > 0
    assert len(all_num_qubits) > 0


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

    # print(fidelities)

    assert len(fidelities) > 0
