import json

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.benchmarks.wit import WITResult
from metriq_gym.exporters.json_exporter import JsonExporter


def test_json_exporter_serializes_none_uncertainty_to_null(metriq_job, tmp_path):
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile) as f:
        data = json.load(f)

    assert data["results"]["expectation_value"]["value"] == 0.42
    assert data["results"]["expectation_value"]["uncertainty"] is None


def test_json_exporter_preserves_device_calibration_metadata(metriq_job, tmp_path):
    metriq_job.platform["device_metadata"] = {
        "num_qubits": 5,
        "calibration": {
            "avg_t1_s": 0.00012,
            "avg_t2_s": 0.00009,
            "avg_readout_error": 0.025,
            "avg_1q_gate_error": 0.001,
            "avg_2q_gate_error": 0.02,
            "last_update_date": "2026-06-01T12:00:00Z",
        },
    }
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile) as f:
        data = json.load(f)

    assert data["platform"]["device_metadata"] == metriq_job.platform["device_metadata"]
