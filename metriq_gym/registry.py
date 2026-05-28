from .constants import JobType

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.benchmarks.qml_kernel import QMLKernel, QMLKernelData, QMLKernelResult
from metriq_gym.benchmarks.clops import Clops, ClopsData, ClopsResult

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
from metriq_gym.benchmarks.mermin import Mermin, MerminData, MerminResult
from metriq_gym.benchmarks.semi_brukner import SemiBrukner, SemiBruknerData, SemiBruknerResult
from metriq_gym.benchmarks.svetlichny import Svetlichny, SvetlichnyData, SvetlichnyResult
from metriq_gym.benchmarks.cglmp import CGLMP, CGLMPData, CGLMPResult
from metriq_gym.benchmarks.magic_square import MagicSquare, MagicSquareData, MagicSquareResult

BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    JobType.BSEQ: BSEQ,
    JobType.CLOPS: Clops,
    JobType.EPLG: EPLG,
    JobType.QML_KERNEL: QMLKernel,
    JobType.MIRROR_CIRCUITS: MirrorCircuits,
    JobType.WIT: WIT,
    JobType.BERNSTEIN_VAZIRANI: QEDCBenchmark,
    JobType.PHASE_ESTIMATION: QEDCBenchmark,
    JobType.HIDDEN_SHIFT: QEDCBenchmark,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCBenchmark,
    JobType.LR_QAOA: LinearRampQAOA,
    JobType.MERMIN: Mermin,
    JobType.SEMI_BRUKNER: SemiBrukner,
    JobType.SVETLICHNY: Svetlichny,
    JobType.CGLMP: CGLMP,
    JobType.MAGIC_SQUARE: MagicSquare,
}

BENCHMARK_DATA_CLASSES: dict[JobType, type[BenchmarkData]] = {
    JobType.BSEQ: BSEQData,
    JobType.CLOPS: ClopsData,
    JobType.EPLG: EPLGData,
    JobType.QML_KERNEL: QMLKernelData,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsData,
    JobType.WIT: WITData,
    JobType.BERNSTEIN_VAZIRANI: QEDCData,
    JobType.PHASE_ESTIMATION: QEDCData,
    JobType.HIDDEN_SHIFT: QEDCData,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCData,
    JobType.LR_QAOA: LinearRampQAOAData,
    JobType.MERMIN: MerminData,
    JobType.SEMI_BRUKNER: SemiBruknerData,
    JobType.SVETLICHNY: SvetlichnyData,
    JobType.CGLMP: CGLMPData,
    JobType.MAGIC_SQUARE: MagicSquareData,
}

BENCHMARK_RESULT_CLASSES: dict[JobType, type[BenchmarkResult]] = {
    JobType.BSEQ: BSEQResult,
    JobType.CLOPS: ClopsResult,
    JobType.EPLG: EPLGResult,
    JobType.QML_KERNEL: QMLKernelResult,
    JobType.MIRROR_CIRCUITS: MirrorCircuitsResult,
    JobType.WIT: WITResult,
    JobType.BERNSTEIN_VAZIRANI: QEDCResult,
    JobType.PHASE_ESTIMATION: QEDCResult,
    JobType.HIDDEN_SHIFT: QEDCResult,
    JobType.QUANTUM_FOURIER_TRANSFORM: QEDCResult,
    JobType.LR_QAOA: LinearRampQAOAResult,
    JobType.MERMIN: MerminResult,
    JobType.SEMI_BRUKNER: SemiBruknerResult,
    JobType.SVETLICHNY: SvetlichnyResult,
    JobType.CGLMP: CGLMPResult,
    JobType.MAGIC_SQUARE: MagicSquareResult,
}


def get_available_benchmarks() -> list[str]:
    return [jt.value for jt in JobType]
