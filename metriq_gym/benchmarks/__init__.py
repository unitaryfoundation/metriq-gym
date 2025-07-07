from enum import StrEnum

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.benchmarks.qml_kernel import QMLKernel, QMLKernelData
from metriq_gym.benchmarks.clops import Clops, ClopsData
from metriq_gym.benchmarks.quantum_volume import QuantumVolume, QuantumVolumeData
from metriq_gym.benchmarks.bseq import BSEQ, BSEQData
from metriq_gym.benchmarks.mirror_circuits import MirrorCircuits, MirrorCircuitsData
from metriq_gym.benchmarks.wormhole import Wormhole, WormholeData
from metriq_gym.benchmarks.qedc_benchmarks import QEDCBenchmark, QEDCData


class JobType(StrEnum):
    BSEQ = "BSEQ"
    CLOPS = "CLOPS"
    QML_KERNEL = "QML Kernel"
    QUANTUM_VOLUME = "Quantum Volume"
    MIRROR_CIRCUITS = "Mirror Circuits"
    WORMHOLE = "Wormhole"
    BERNSTEIN_VAZIRANI = "Bernstein-Vazirani"
    PHASE_ESTIMATION = "Phase Estimation"
    HIDDEN_SHIFT = "Hidden Shift"
    QUANTUM_FOURIER_TRANSFORM = "Quantum Fourier Transform"


BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    JobType.BSEQ: BSEQ,
    JobType.CLOPS: Clops,
    JobType.QML_KERNEL: QMLKernel,
    JobType.QUANTUM_VOLUME: QuantumVolume,
    JobType.MIRROR_CIRCUITS: MirrorCircuits,
    JobType.WORMHOLE: Wormhole,
    JobType.BERNSTEIN_VAZIRANI: QEDCBenchmark,
    JobType.PHASE_ESTIMATION: QEDCBenchmark,
    JobType.HIDDEN_SHIFT: QEDCBenchmark,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCBenchmark,
}

BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
    JobType.BSEQ: BSEQData,
    JobType.CLOPS: ClopsData,
    JobType.QML_KERNEL: QMLKernelData,
    JobType.QUANTUM_VOLUME: QuantumVolumeData,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsData,
    JobType.WORMHOLE: WormholeData,
    JobType.BERNSTEIN_VAZIRANI: QEDCData,
    JobType.PHASE_ESTIMATION: QEDCData,
    JobType.HIDDEN_SHIFT: QEDCData,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCData,
}

SCHEMA_MAPPING = {
    JobType.BSEQ: "bseq.schema.json",
    JobType.CLOPS: "clops.schema.json",
    JobType.QML_KERNEL: "qml_kernel.schema.json",
    JobType.QUANTUM_VOLUME: "quantum_volume.schema.json",
    JobType.MIRROR_CIRCUITS: "mirror_circuits.schema.json",
    JobType.WORMHOLE: "wormhole.schema.json",
    JobType.BERNSTEIN_VAZIRANI: "bernstein_vazirani.schema.json",
    JobType.PHASE_ESTIMATION: "phase_estimation.schema.json",
    JobType.HIDDEN_SHIFT: "hidden_shift.schema.json",
    JobType.QUANTUM_FOURIER_TRANSFORM: "quantum_fourier_transform.schema.json",
}


def get_available_benchmarks() -> list[str]:
    return [jt.value for jt in JobType]
