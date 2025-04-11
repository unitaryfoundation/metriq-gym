from enum import StrEnum


class JobType(StrEnum):
    BSEQ = "BSEQ"
    CLOPS = "CLOPS"
    QML_KERNEL = "QML Kernel"
    QUANTUM_VOLUME = "Quantum Volume"
    WORMHOLE = "Wormhole"
