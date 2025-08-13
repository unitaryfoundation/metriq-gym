from .constants import JobType

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.benchmarks.qml_kernel import QMLKernel, QMLKernelData, QMLKernelResult
from metriq_gym.benchmarks.bseq import BSEQResult
from metriq_gym.benchmarks.clops import ClopsResult
from metriq_gym.benchmarks.quantum_volume import QuantumVolumeResult
from metriq_gym.benchmarks.mirror_circuits import MirrorCircuitsResult
from metriq_gym.benchmarks.wormhole import WormholeResult
from metriq_gym.benchmarks.qedc_benchmarks import QEDCResult
from metriq_gym.benchmarks.clops import Clops, ClopsData
from metriq_gym.benchmarks.quantum_volume import QuantumVolume, QuantumVolumeData
from metriq_gym.benchmarks.bseq import BSEQ, BSEQData
from metriq_gym.benchmarks.mirror_circuits import MirrorCircuits, MirrorCircuitsData
from metriq_gym.benchmarks.wormhole import Wormhole, WormholeData
from metriq_gym.benchmarks.qedc_benchmarks import QEDCBenchmark, QEDCData

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

BENCHMARK_RESULT_CLASSES: dict[JobType, type[BenchmarkResult]] = {
    JobType.BSEQ: BSEQResult,
    JobType.CLOPS: ClopsResult,
    JobType.QML_KERNEL: QMLKernelResult,
    JobType.QUANTUM_VOLUME: QuantumVolumeResult,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsResult,
    JobType.WORMHOLE: WormholeResult,
    JobType.BERNSTEIN_VAZIRANI: QEDCResult,
    JobType.PHASE_ESTIMATION: QEDCResult,
    JobType.HIDDEN_SHIFT: QEDCResult,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCResult,
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
