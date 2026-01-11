from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.benchmarks.wit import WITResult
from metriq_gym.exporters.cli_exporter import CliExporter


def test_cli_exporter_displays_value_with_uncertainty(metriq_job, capsys):
    result = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))

    CliExporter(metriq_job, result).export()

    output = capsys.readouterr().out
    assert "expectation_value: 0.5 ± 0.05" in output


def test_cli_exporter_omits_uncertainty_when_missing(metriq_job, capsys):
    # When uncertainty is not provided, it should default to None and not be printed
    result = WITResult(expectation_value=BenchmarkScore(value=0.5))

    CliExporter(metriq_job, result).export()

    output = capsys.readouterr().out
    assert "expectation_value: 0.5" in output
    assert "±" not in output
