import os
import json

from qiskit import QuantumCircuit

from metriq_gym.local.aer import (
    AerSimulatorDevice,
    JOB_STORAGE_FILE,
    load_local_job,
)
from metriq_gym.qplatform.device import connectivity_graph
import rustworkx as rx


def test_aer_simulator_device_run(tmp_path):
    from metriq_gym.local.aer import AerSimulatorDevice

    os.chdir(tmp_path)

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    device = AerSimulatorDevice()
    job_type = "quantum-volume-test"
    job = device.run(qc, shots=100, job_type=job_type)
    result = job.result()

    assert isinstance(result.data.measurement_counts, dict)
    assert os.path.exists(JOB_STORAGE_FILE)

    # Validate stored content
    with open(JOB_STORAGE_FILE) as f:
        lines = f.readlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["id"] == job.id
    assert entry["device_name"] == "aer_simulator"
    assert entry["job_type"] == job_type
    assert "measurement_counts" in entry["data"]

    # Check reloading
    reloaded = load_local_job(job.id)
    assert reloaded.id == job.id
    assert reloaded.job_type == job_type
    assert reloaded.result().data.measurement_counts == result.data.measurement_counts


def test_aer_simulator_connectivity():
    device = AerSimulatorDevice()
    graph = connectivity_graph(device)

    assert isinstance(graph, rx.PyGraph)
    assert graph.num_nodes() == device.backend.num_qubits
