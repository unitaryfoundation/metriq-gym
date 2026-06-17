import json

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.benchmarks.wit import WITResult
from metriq_gym.exporters.json_exporter import JsonExporter


def test_json_exporter_serializes_none_uncertainty_to_null(metriq_job, tmp_path):
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile, encoding="utf-8") as f:
        data = json.load(f)

    assert data["results"]["expectation_value"]["value"] == 0.42
    assert data["results"]["expectation_value"]["uncertainty"] is None


def test_json_exporter_includes_two_qubit_gate_counts(metriq_job, tmp_path):
    metriq_job.data["input_two_qubit_gate_counts"] = [2, 4]
    metriq_job.data["transpiled_two_qubit_gate_counts"] = [3, 5]
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile, encoding="utf-8") as f:
        data = json.load(f)

    assert data["input_two_qubit_gate_counts"] == [2, 4]
    assert data["transpiled_two_qubit_gate_counts"] == [3, 5]
