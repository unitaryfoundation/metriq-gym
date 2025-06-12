import json
import os
import uuid
from datetime import datetime

from qbraid.runtime import JobStatus, GateModelResultData
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

JOB_STORAGE_FILE = ".metriq_gym_jobs.jsonl"
os.makedirs(os.path.dirname(JOB_STORAGE_FILE) or ".", exist_ok=True)


def _append_job_to_file(job_dict: dict) -> None:
    with open(JOB_STORAGE_FILE, "a") as f:
        f.write(json.dumps(job_dict) + "\n")


def _load_job_from_file(job_id: str) -> dict:
    if not os.path.exists(JOB_STORAGE_FILE):
        return {}
    with open(JOB_STORAGE_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("id") == job_id:
                return entry
    return {}


class LocalResult:
    def __init__(self, data: dict):
        self.data = GateModelResultData(measurement_counts=data.get("measurement_counts", {}))


class LocalJob:
    def __init__(
        self,
        job_id: str,
        counts: dict[str, int],
        job_type: str = "local",
        params: dict = {},
        data: dict = {},
        provider_name: str = "local",
        device_name: str = "aer_simulator",
        dispatch_time: str = "",
    ):
        self.id = job_id
        self._counts = counts
        self.job_type = job_type
        self.params = params
        self.data = data
        self.provider_name = provider_name
        self.device_name = device_name
        self.dispatch_time = dispatch_time or datetime.now().isoformat()
        self._save()

    def _save(self) -> None:
        job_entry = {
            "id": self.id,
            "job_type": self.job_type,
            "params": self.params,
            "data": {"measurement_counts": self._counts} | self.data,
            "provider_name": self.provider_name,
            "device_name": self.device_name,
            "dispatch_time": self.dispatch_time,
            "result_data": None,
        }
        _append_job_to_file(job_entry)

    def status(self) -> JobStatus:
        return JobStatus.COMPLETED

    def result(self) -> LocalResult:
        return LocalResult({"measurement_counts": self._counts})


class AerSimulatorDevice:
    id = "aer_simulator"

    def __init__(self) -> None:
        self.backend = AerSimulator()
        self.num_qubits = self.backend.configuration().n_qubits

    def run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit],
        shots: int | None = None,
        job_type: str = "local",
        params: dict = {},
        data: dict = {},
    ) -> LocalJob | list[LocalJob]:
        circ_list = circuits if isinstance(circuits, list) else [circuits]
        jobs = []
        for circ in circ_list:
            aer_job = self.backend.run(circ, shots=shots)
            result = aer_job.result()
            counts = result.get_counts(circ)
            jobs.append(
                LocalJob(
                    job_id=str(uuid.uuid4()),
                    counts=counts,
                    job_type=job_type,
                    params=params,
                    data=data | {"shots": shots or 1024},
                )
            )
        return jobs[0] if isinstance(circuits, QuantumCircuit) else jobs


def load_local_job(job_id: str) -> LocalJob:
    job_dict = _load_job_from_file(job_id)
    if not job_dict:
        raise FileNotFoundError(f"No job with id {job_id} found in {JOB_STORAGE_FILE}")
    counts = job_dict.get("data", {}).get("measurement_counts", {})
    return LocalJob(
        job_id=job_dict["id"],
        counts=counts,
        job_type=job_dict.get("job_type", "local"),
        params=job_dict.get("params", {}),
        data=job_dict.get("data", {}),
        provider_name=job_dict.get("provider_name", "local"),
        device_name=job_dict.get("device_name", "aer_simulator"),
        dispatch_time=job_dict.get("dispatch_time", ""),
    )


class LocalDevice:
    def run(self, shots: int) -> LocalJob:
        job = LocalJob(str(uuid.uuid4()), {"shots": shots})
        job._save()
        return job
