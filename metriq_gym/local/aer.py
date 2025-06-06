import json
import os
import uuid
from uuid import uuid4

from qbraid.runtime import JobStatus, GateModelResultData
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


LOCAL_JOB_DIR = ".metriq_local_jobs"
os.makedirs(LOCAL_JOB_DIR, exist_ok=True)


class LocalResult:
    def __init__(self, counts: dict[str, int]):
        self.data = GateModelResultData(measurement_counts=counts)


class LocalJob:
    def __init__(self, job_id: str, counts: dict[str, int]):
        self.id = job_id
        self._counts = counts
        self._save()

    def _save(self) -> None:
        path = os.path.join(LOCAL_JOB_DIR, f"{self.id}.json")
        with open(path, "w") as f:
            json.dump({"measurement_counts": self._counts}, f)

    def status(self) -> JobStatus:
        return JobStatus.COMPLETED

    def result(self) -> LocalResult:
        return LocalResult(self._counts)


class AerSimulatorDevice:
    id = "aer_simulator"

    def __init__(self) -> None:
        self.backend = AerSimulator()
        self.num_qubits = self.backend.configuration().n_qubits

    def run(
        self, circuits: QuantumCircuit | list[QuantumCircuit], shots: int | None = None
    ) -> LocalJob | list[LocalJob]:
        circ_list = circuits if isinstance(circuits, list) else [circuits]
        jobs = []
        for circ in circ_list:
            aer_job = self.backend.run(circ, shots=shots)
            result = aer_job.result()
            counts = result.get_counts(circ)
            jobs.append(LocalJob(str(uuid.uuid4()), counts))
        return jobs[0] if isinstance(circuits, QuantumCircuit) else jobs


def load_local_job(job_id: str) -> LocalJob:
    path = os.path.join(LOCAL_JOB_DIR, f"{job_id}.json")
    with open(path) as f:
        data = json.load(f)
    return LocalJob(job_id, data["measurement_counts"])


class LocalDevice:
    def run(self, shots: int | None = None) -> LocalJob:
        job = LocalJob(str(uuid.uuid4()), {"shots": shots})
        job._save()
        return job