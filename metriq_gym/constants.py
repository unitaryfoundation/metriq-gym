from enum import StrEnum


class JobType(StrEnum):
    BSEQ = "BSEQ"
    CLOPS = "CLOPS"
    EPLG = "EPLG"
    QML_KERNEL = "QML Kernel"
    QUANTUM_VOLUME = "Quantum Volume"
    MIRROR_CIRCUITS = "Mirror Circuits"
    WIT = "WIT"
    BERNSTEIN_VAZIRANI = "Bernstein-Vazirani"
    PHASE_ESTIMATION = "Phase Estimation"
    HIDDEN_SHIFT = "Hidden Shift"
    QUANTUM_FOURIER_TRANSFORM = "Quantum Fourier Transform"
    LR_QAOA = "Linear Ramp QAOA"


SCHEMA_MAPPING = {
    JobType.BSEQ: "bseq.schema.json",
    JobType.CLOPS: "clops.schema.json",
    JobType.EPLG: "eplg.schema.json",
    JobType.QML_KERNEL: "qml_kernel.schema.json",
    JobType.QUANTUM_VOLUME: "quantum_volume.schema.json",
    JobType.MIRROR_CIRCUITS: "mirror_circuits.schema.json",
    JobType.WIT: "wit.schema.json",
    JobType.BERNSTEIN_VAZIRANI: "bernstein_vazirani.schema.json",
    JobType.PHASE_ESTIMATION: "phase_estimation.schema.json",
    JobType.HIDDEN_SHIFT: "hidden_shift.schema.json",
    JobType.QUANTUM_FOURIER_TRANSFORM: "quantum_fourier_transform.schema.json",
    JobType.LR_QAOA: "lr_qaoa.schema.json",
}
