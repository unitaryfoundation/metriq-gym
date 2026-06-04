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


def test_json_exporter_preserves_nested_device_calibration_metadata(metriq_job, tmp_path):
    metriq_job.platform = {
        "provider": "ibm",
        "device": "ibm_sherbrooke",
        "device_metadata": {
            "version": "1.6.73",
            "calibration": {
                "avg_t1_s": 0.00012,
                "avg_t2_s": 0.00009,
                "avg_readout_error": 0.021,
                "avg_1q_gate_error": 0.0004,
                "avg_2q_gate_error": 0.008,
                "last_update_date": "2026-01-16T15:30:00+00:00",
            },
        },
    }
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile) as f:
        data = json.load(f)

    assert data["platform"]["device_metadata"]["calibration"] == {
        "avg_t1_s": 0.00012,
        "avg_t2_s": 0.00009,
        "avg_readout_error": 0.021,
        "avg_1q_gate_error": 0.0004,
        "avg_2q_gate_error": 0.008,
        "last_update_date": "2026-01-16T15:30:00+00:00",
    }
