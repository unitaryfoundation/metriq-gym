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
