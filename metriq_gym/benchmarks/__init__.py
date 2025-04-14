from enum import StrEnum

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.benchmarks.qml_kernel import QMLKernel, QMLKernelData
from metriq_gym.benchmarks.clops import Clops, ClopsData
from metriq_gym.benchmarks.quantum_volume import QuantumVolume, QuantumVolumeData
from metriq_gym.benchmarks.bseq import BSEQ, BSEQData


class JobType(StrEnum):
    BSEQ = "BSEQ"
    CLOPS = "CLOPS"
    QML_KERNEL = "QML Kernel"
    QUANTUM_VOLUME = "Quantum Volume"


BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    JobType.BSEQ: BSEQ,
    JobType.CLOPS: Clops,
    JobType.QML_KERNEL: QMLKernel,
    JobType.QUANTUM_VOLUME: QuantumVolume,
}

BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
    JobType.BSEQ: BSEQData,
    JobType.CLOPS: ClopsData,
    JobType.QML_KERNEL: QMLKernelData,
    JobType.QUANTUM_VOLUME: QuantumVolumeData,
}

SCHEMA_MAPPING = {
    JobType.BSEQ: "bseq.schema.json",
    JobType.CLOPS: "clops.schema.json",
    JobType.QML_KERNEL: "qml_kernel.schema.json",
    JobType.QUANTUM_VOLUME: "quantum_volume.schema.json",
}
