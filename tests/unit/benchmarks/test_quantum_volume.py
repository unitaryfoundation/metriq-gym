import pytest
from metriq_gym.benchmarks.quantum_volume import calc_stats, QuantumVolumeData, prepare_qv_circuits


@pytest.mark.parametrize("n, trials", [(2, 2), (3, 3)])
def test_prepare_qv_circuits(n, trials):
    circuits, ideal_probs = prepare_qv_circuits(n=n, num_trials=trials)
    assert len(circuits) == trials
    assert len(ideal_probs) == trials
    assert circuits[0].size() == ((n // 2) + n + 1) * n  # (cx + u3 + measure) * depth = num_qubits
    assert len(ideal_probs[0]) == 2**n


def test_calc_stats_pass():
    job_data = QuantumVolumeData(
        provider_job_ids=["test_job_id"],
        num_qubits=2,
        shots=100,
        depth=2,
        confidence_level=0.7,
        ideal_probs=[
            [0.8, 0.2, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
        ],
        trials=2,
    )
    counts = [
        {"00": 80, "01": 20, "10": 0, "11": 0},
        {"00": 60, "01": 40, "10": 0, "11": 0},
    ]
    stats = calc_stats(job_data, counts)
    assert stats.confidence_pass is True  # All trials pass confidence level


def test_calc_stats_not_pass():
    job_data = QuantumVolumeData(
        provider_job_ids=["test_job_id"],
        num_qubits=2,
        shots=100,
        depth=2,
        confidence_level=0.7,
        ideal_probs=[
            [0.8, 0.2, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
        ],
        trials=2,
    )
    counts = [
        {"00": 10, "01": 20, "10": 50, "11": 20},
        {"00": 10, "01": 40, "10": 10, "11": 40},
    ]
    stats = calc_stats(job_data, counts)
    assert stats.confidence_pass is False  # Not all trials pass confidence level
