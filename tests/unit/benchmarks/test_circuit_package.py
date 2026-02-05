"""Tests for CircuitPackage and supports_qem flag on benchmarks."""

import pytest
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import CircuitPackage, Benchmark


class TestCircuitPackage:
    def test_creation(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        pkg = CircuitPackage(circuits=[qc], shots=1000)
        assert len(pkg.circuits) == 1
        assert pkg.shots == 1000
        assert pkg.metadata == {}

    def test_with_metadata(self):
        qc = QuantumCircuit(1)
        pkg = CircuitPackage(
            circuits=[qc],
            shots=500,
            metadata={"expected": "01", "width": 4},
        )
        assert pkg.metadata["expected"] == "01"
        assert pkg.metadata["width"] == 4


class TestSupportsQem:
    def test_base_class_default_false(self):
        assert Benchmark.supports_qem is False

    def test_compatible_benchmarks(self):
        """All benchmarks marked supports_qem=True should be importable."""
        from metriq_gym.benchmarks.mirror_circuits import MirrorCircuits
        from metriq_gym.benchmarks.quantum_volume import QuantumVolume
        from metriq_gym.benchmarks.wit import WIT
        from metriq_gym.benchmarks.qml_kernel import QMLKernel
        from metriq_gym.benchmarks.qedc_benchmarks import QEDCBenchmark
        from metriq_gym.benchmarks.lr_qaoa import LinearRampQAOA

        assert MirrorCircuits.supports_qem is True
        assert QuantumVolume.supports_qem is True
        assert WIT.supports_qem is True
        assert QMLKernel.supports_qem is True
        assert QEDCBenchmark.supports_qem is True
        assert LinearRampQAOA.supports_qem is True

    def test_incompatible_benchmarks(self):
        from metriq_gym.benchmarks.clops import Clops
        from metriq_gym.benchmarks.bseq import BSEQ
        from metriq_gym.benchmarks.eplg import EPLG

        assert Clops.supports_qem is False
        assert BSEQ.supports_qem is False
        assert EPLG.supports_qem is False
