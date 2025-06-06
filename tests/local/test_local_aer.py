import os
import json
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from metriq_gym.local.aer import (
    AerSimulatorDevice,
    load_local_job,
    LOCAL_JOB_DIR,
    LocalDevice,
)
from metriq_gym.qplatform.device import connectivity_graph
import rustworkx as rx

def test_aer_simulator_device_run(tmp_path):
    os.chdir(tmp_path)
    os.makedirs(LOCAL_JOB_DIR, exist_ok=True)

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    backend = AerSimulator()
    job = backend.run(qc)
    result = job.result()

    assert result.success
    assert (tmp_path / LOCAL_JOB_DIR).exists()

def test_aer_simulator_connectivity():
    device = AerSimulatorDevice()
    graph = connectivity_graph(device)

    assert isinstance(graph, rx.PyGraph)
    assert graph.num_nodes() == device.backend.num_qubits

def test_run_saves_job_in_cwd(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        device = LocalDevice()
        job = device.run(None, shots=10)
        job_file = tmp_path / LOCAL_JOB_DIR / f"{job.job_id}.json"

        assert job_file.exists()

        with open(job_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["shots"] == 10

    finally:
        os.chdir(cwd)
