"""QSC benchmark: dynamical structure factor via a Ramsey-type protocol.

Summary:
    Drives the QSC-Applications data-collection protocol through metriq-gym's
    provider abstraction. For a 1-D transverse-field Ising chain, the protocol
    prepares the ground state, applies a single-site Pauli perturbation, evolves
    under the Hamiltonian, and measures every site in a chosen Pauli basis. That
    sequence is repeated over a sweep of evolution times and over perturbation
    and measurement basis pairs, which together give the unequal-time correlation
    functions needed for the retarded Green's function.

    A single dispatch therefore submits a *batch* of circuits, not one circuit.
    For the center-site experiment class the batch is 12 circuits per time step
    (3 unperturbed plus 9 perturbation/measurement pairs).

Result interpretation:
    Polling returns the retarded Green's function G^R(i, j, t) with propagated
    uncertainties, as computed by QSC-Applications' own post-processing. The
    device output is not scored on its own: scoring requires a reference to
    compare against, which is the validation step that is not built yet. See
    `compute_score` below.

References:
    - Circuit construction and post-processing: QSC-Applications, branch
      release/v0.1, vendored at submodules/QSC-Applications.
    - Protocol: Baez et al., "Dynamical structure factors of dynamically
      generated quantum states", PNAS 117(42), 2020.
      https://doi.org/10.1073/pnas.2006103117
"""

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


# ---------------------------------------------------------------------------
# QSC-Applications loader
# ---------------------------------------------------------------------------

# QSC-Applications is not a package: its modules import each other as
# ``from src...``, so the submodule root has to be importable as a top-level
# path. It is inserted lazily rather than at metriq-gym import time so that the
# generic name ``src`` only enters sys.modules for runs that actually use QSC.
QSC_ROOT = Path(__file__).resolve().parents[2] / "submodules" / "QSC-Applications"


def _load_qsc() -> tuple[ModuleType, ModuleType]:
    """Import the QSC-Applications facade and post-processing modules."""
    if not (QSC_ROOT / "src" / "experiment_facade.py").exists():
        raise RuntimeError(
            f"QSC-Applications not found at {QSC_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    if str(QSC_ROOT) not in sys.path:
        sys.path.insert(0, str(QSC_ROOT))

    from src import experiment_facade  # noqa: PLC0415
    from src.utils import postprocessing_utils  # noqa: PLC0415

    return experiment_facade, postprocessing_utils


# ---------------------------------------------------------------------------
# Benchmark classes
# ---------------------------------------------------------------------------


class QSCResult(BenchmarkResult):
    """Retarded Green's function measured on the device.

    Attributes:
        green_function: G^R(i, j, t) for each (time_step, i, j), with the
            uncertainty propagated from the unequal-time correlators.
        time_steps: Number of evolution times swept.
        sites: Number of lattice sites.
        perturbation_basis: Pauli basis of the applied perturbation (alpha).
        measurement_basis: Pauli basis measured (beta).
    """

    green_function: list[BenchmarkScore]
    time_steps: int
    sites: int
    perturbation_basis: str
    measurement_basis: str

    def compute_score(self) -> BenchmarkScore | None:
        """No scalar score yet.

        The QSC benchmark is a validation measurement, not a figure of merit:
        G^R only means something relative to a reference computed for the same
        Hamiltonian, time, and basis. The intended score is a reduced chi-squared
        against that reference, which is not wired up yet. Returning None rather
        than an arbitrary summary keeps an uninterpretable number out of
        metriq-data.
        """
        return None


@dataclass
class QSCData(BenchmarkData):
    """Data carried between dispatch and poll."""

    sites: int = 0
    shots: int = 0
    time_steps: int = 0
    experiment_class: str = "DFS-CS"
    perturbation_basis: str = "X"
    measurement_basis: str = "X"
    circuit_names: list[str] = field(default_factory=list)


class QSC(Benchmark):
    """QSC dynamical structure factor benchmark."""

    def _facade(self):
        """Construct the QSC data-collection facade from the job config."""
        facade_module, _ = _load_qsc()
        params = self.params

        coefs = dict(params.coefs)
        simulation = facade_module.QiskitDataCollectionDFS(
            hamiltonian_kwargs={
                "name": params.hamiltonian,
                "sites": params.num_qubits,
                "pauli_terms": list(params.pauli_terms),
                "coefs": coefs,
                "is_periodic": params.is_periodic,
            },
            state_prep_kwargs={"sites": params.num_qubits, "spec": "ground_state"},
            dynamics_kwargs=self._dynamics_kwargs(),
            implementation_kwargs={"shots": params.shots},
        )
        return simulation

    def _dynamics_kwargs(self) -> dict[str, Any]:
        params = self.params
        if params.dynamics == "exact_dynamics":
            return {"synth_mode": "exact_dynamics"}
        # Note: the facade reads 'reps', while its docstring and the bundled
        # runscripts say 'steps'. Passing 'steps' silently leaves reps at 1.
        return {
            "synth_mode": "approx_dynamics",
            "approx_name": params.approx_name,
            "order": params.trotter_order,
            "reps": params.trotter_reps,
        }

    def _build_circuits(self, device: "QuantumDevice") -> list[QuantumCircuit]:
        simulation = self._facade()
        return simulation.build_experiments(
            experiment_kwargs={
                "experiment_class": self.params.experiment_class,
                "strength": self.params.strength,
                "time_steps": self.params.time_steps,
                "output_dir": self.params.output_dir,
            }
        )

    def dispatch_handler(self, device: "QuantumDevice") -> QSCData:
        circuits = self._build_circuits(device)
        quantum_job = device.run(circuits, shots=self.params.shots)
        return QSCData.from_quantum_job(
            quantum_job,
            sites=self.params.num_qubits,
            shots=self.params.shots,
            time_steps=self.params.time_steps,
            experiment_class=self.params.experiment_class,
            perturbation_basis=self.params.perturbation_basis,
            measurement_basis=self.params.measurement_basis,
            circuit_names=[circuit.name for circuit in circuits],
        )

    def poll_handler(
        self,
        job_data: QSCData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QSCResult:
        counts_list = flatten_counts(result_data)
        if len(counts_list) != len(job_data.circuit_names):
            raise ValueError(
                f"QSC expected {len(job_data.circuit_names)} count sets, got {len(counts_list)}"
            )

        _, postprocessing = _load_qsc()

        # QSC's post-processing keys off the circuit name, which encodes the
        # perturbed site, the perturbation and measurement bases, the strength,
        # and the time-step index.
        records = [
            [index, name, "metriq-gym", {key: int(value) for key, value in counts.items()}]
            for index, (name, counts) in enumerate(zip(job_data.circuit_names, counts_list))
        ]

        # compute_rgf reads the counts back from a file rather than taking them
        # directly, so round-trip through a temporary one.
        with tempfile.TemporaryDirectory() as workdir:
            results_path = Path(workdir) / "results.json"
            results_path.write_text(json.dumps(records), encoding="utf-8")
            frame = postprocessing.compute_rgf(
                results_filepath=str(results_path),
                alpha=job_data.perturbation_basis,
                beta=job_data.measurement_basis,
            )

        green_function = [
            BenchmarkScore(value=float(row.RGF), uncertainty=float(row.RGF_err))
            for row in frame.itertuples()
        ]
        return QSCResult(
            green_function=green_function,
            time_steps=job_data.time_steps,
            sites=job_data.sites,
            perturbation_basis=job_data.perturbation_basis,
            measurement_basis=job_data.measurement_basis,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
