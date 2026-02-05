"""Tests for QEM dispatch and poll integration helpers."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from metriq_gym.qem.dispatch import dispatch_with_qem
from metriq_gym.qem.poll import poll_with_qem, _extract_unscaled_counts


class TestDispatchWithQem:
    def test_dispatch_calls_build_circuits(self):
        """dispatch_with_qem should call handler.build_circuits, not dispatch_handler."""
        from metriq_gym.benchmarks.benchmark import CircuitPackage, BenchmarkData
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        mock_handler = MagicMock()
        mock_handler.build_circuits.return_value = CircuitPackage(
            circuits=[qc], shots=100, metadata={}
        )

        @dataclass
        class MockData(BenchmarkData):
            pass

        mock_handler.create_job_data.return_value = MockData(
            provider_job_ids=["job-1"]
        )

        mock_device = MagicMock()
        mock_device.run.return_value = MagicMock(id="job-1")

        qem_config = [
            {"technique": "zne", "scale_factors": [1.0, 2.0, 3.0]}
        ]

        job_data, qem_data = dispatch_with_qem(mock_handler, mock_device, qem_config)

        mock_handler.build_circuits.assert_called_once_with(mock_device)
        mock_handler.dispatch_handler.assert_not_called()
        mock_handler.create_job_data.assert_called_once()

        # Device should receive 3 circuits (1 * 3 scale factors)
        call_args = mock_device.run.call_args
        circuits_submitted = call_args[0][0]
        assert len(circuits_submitted) == 3

    def test_qem_data_structure(self):
        """qem_data should contain config, metadata, and original_circuit_count."""
        from metriq_gym.benchmarks.benchmark import CircuitPackage, BenchmarkData
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        mock_handler = MagicMock()
        mock_handler.build_circuits.return_value = CircuitPackage(
            circuits=[qc, qc], shots=100, metadata={}
        )

        @dataclass
        class MockData(BenchmarkData):
            pass

        mock_handler.create_job_data.return_value = MockData(
            provider_job_ids=["job-1"]
        )
        mock_device = MagicMock()
        mock_device.run.return_value = MagicMock(id="job-1")

        _, qem_data = dispatch_with_qem(
            mock_handler, mock_device,
            [{"technique": "zne", "scale_factors": [1.0, 2.0]}],
        )

        assert "config" in qem_data
        assert "metadata" in qem_data
        assert qem_data["original_circuit_count"] == 2
        assert qem_data["config"] == [
            {"technique": "zne", "scale_factors": [1.0, 2.0]}
        ]
        assert len(qem_data["metadata"]) == 1
        assert qem_data["metadata"][0]["technique_name"] == "zne"


class TestExtractUnscaledCounts:
    def test_basic_extraction(self):
        """Should extract the unscaled (1.0) counts from interleaved results."""
        all_counts = [
            {"0": 90, "1": 10},   # circuit 0, scale 1.0
            {"0": 70, "1": 30},   # circuit 0, scale 2.0
            {"0": 50, "1": 50},   # circuit 0, scale 3.0
            {"0": 80, "1": 20},   # circuit 1, scale 1.0
            {"0": 60, "1": 40},   # circuit 1, scale 2.0
            {"0": 40, "1": 60},   # circuit 1, scale 3.0
        ]
        qem_data = {
            "original_circuit_count": 2,
            "metadata": [{
                "technique_name": "zne",
                "data": {"scale_factors": [1.0, 2.0, 3.0]},
            }],
        }
        raw = _extract_unscaled_counts(all_counts, qem_data)
        assert len(raw) == 2
        assert raw[0] == {"0": 90, "1": 10}
        assert raw[1] == {"0": 80, "1": 20}

    def test_no_metadata(self):
        """With no metadata, should return all counts as-is."""
        all_counts = [{"0": 50, "1": 50}]
        qem_data = {
            "original_circuit_count": 1,
            "metadata": [],
        }
        raw = _extract_unscaled_counts(all_counts, qem_data)
        assert raw == all_counts

    def test_technique_without_scale_factors(self):
        """Techniques without scale_factors should return all counts."""
        all_counts = [{"0": 50, "1": 50}]
        qem_data = {
            "original_circuit_count": 1,
            "metadata": [{
                "technique_name": "dd",
                "data": {},
            }],
        }
        raw = _extract_unscaled_counts(all_counts, qem_data)
        assert raw == all_counts
