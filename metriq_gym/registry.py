from .constants import JobType

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.benchmarks.qml_kernel import QMLKernel, QMLKernelData, QMLKernelResult
from metriq_gym.benchmarks.clops import Clops, ClopsData, ClopsResult
from metriq_gym.benchmarks.quantum_volume import (
    QuantumVolume,
    QuantumVolumeData,
    QuantumVolumeResult,
)
from metriq_gym.benchmarks.bseq import BSEQ, BSEQData, BSEQResult
from metriq_gym.benchmarks.mirror_circuits import (
    MirrorCircuits,
    MirrorCircuitsData,
    MirrorCircuitsResult,
)
from metriq_gym.benchmarks.wit import WIT, WITData, WITResult
from metriq_gym.benchmarks.qedc_benchmarks import QEDCBenchmark, QEDCData, QEDCResult
from metriq_gym.benchmarks.lr_qaoa import LinearRampQAOA, LinearRampQAOAData, LinearRampQAOAResult
from metriq_gym.benchmarks.eplg import EPLG, EPLGData, EPLGResult

BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    JobType.BSEQ: BSEQ,
    JobType.CLOPS: Clops,
    JobType.EPLG: EPLG,
    JobType.QML_KERNEL: QMLKernel,
    JobType.QUANTUM_VOLUME: QuantumVolume,
    JobType.MIRROR_CIRCUITS: MirrorCircuits,
    JobType.WIT: WIT,
    JobType.BERNSTEIN_VAZIRANI: QEDCBenchmark,
    JobType.PHASE_ESTIMATION: QEDCBenchmark,
    JobType.HIDDEN_SHIFT: QEDCBenchmark,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCBenchmark,
    JobType.LR_QAOA: LinearRampQAOA,
}

BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
    JobType.BSEQ: BSEQData,
    JobType.CLOPS: ClopsData,
    JobType.EPLG: EPLGData,
    JobType.QML_KERNEL: QMLKernelData,
    JobType.QUANTUM_VOLUME: QuantumVolumeData,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsData,
    JobType.WIT: WITData,
    JobType.BERNSTEIN_VAZIRANI: QEDCData,
    JobType.PHASE_ESTIMATION: QEDCData,
    JobType.HIDDEN_SHIFT: QEDCData,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCData,
    JobType.LR_QAOA: LinearRampQAOAData,
}

BENCHMARK_RESULT_CLASSES: dict[JobType, type[BenchmarkResult]] = {
    JobType.BSEQ: BSEQResult,
    JobType.CLOPS: ClopsResult,
    JobType.EPLG: EPLGResult,
    JobType.QML_KERNEL: QMLKernelResult,
    JobType.QUANTUM_VOLUME: QuantumVolumeResult,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsResult,
    JobType.WIT: WITResult,
    JobType.BERNSTEIN_VAZIRANI: QEDCResult,
    JobType.PHASE_ESTIMATION: QEDCResult,
    JobType.HIDDEN_SHIFT: QEDCResult,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCResult,
    JobType.LR_QAOA: LinearRampQAOAResult,
}


def get_available_benchmarks() -> list[str]:
    return [jt.value for jt in JobType]
