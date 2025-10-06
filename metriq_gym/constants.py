from enum import StrEnum


class JobType(StrEnum):
    BSEQ = "BSEQ"
    CLOPS = "CLOPS"
    QML_KERNEL = "QML Kernel"
    QUANTUM_VOLUME = "Quantum Volume"
    MIRROR_CIRCUITS = "Mirror Circuits"
    WIT = "WIT"
    BERNSTEIN_VAZIRANI = "Bernstein-Vazirani"
    PHASE_ESTIMATION = "Phase Estimation"
    HIDDEN_SHIFT = "Hidden Shift"
    QUANTUM_FOURIER_TRANSFORM = "Quantum Fourier Transform"
    LR_QAOA = "Linear Ramp QAOA"
    EPLG = "EPLG"
