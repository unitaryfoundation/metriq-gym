import pytest
import json
from pathlib import Path
from metriq_gym.suite_parser import BenchmarkEntry, Suite, parse_suite_file


def test_benchmark_entry_model():
    entry = BenchmarkEntry(name="test_benchmark", config={"param": 1})
    assert entry.name == "test_benchmark"
    assert entry.config == {"param": 1}
    # Defaults
    assert entry.weight == 1.0


def test_suite_model():
    entry1 = BenchmarkEntry(name="b1", config={"a": 1}, weight=0.6)
    entry2 = BenchmarkEntry(name="b2", config={"b": 2}, weight=0.4)
    suite = Suite(name="suite1", benchmarks=[entry1, entry2])
    assert suite.name == "suite1"
    assert len(suite.benchmarks) == 2
    assert suite.benchmarks[0].name == "b1"
    assert suite.benchmarks[1].config == {"b": 2}
    assert suite.benchmarks[0].weight == 0.6


def test_parse_suite_file(tmp_path):
    suite_data = {
        "name": "suite_test",
        "benchmarks": [
            {"name": "bench1", "config": {"x": 10}, "weight": 0.7},
            {"name": "bench2", "config": {"y": 20}, "weight": 0.3},
        ],
    }
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps(suite_data))
    suite = parse_suite_file(suite_file)
    assert suite.name == "suite_test"
    assert len(suite.benchmarks) == 2
    assert suite.benchmarks[0].name == "bench1"
    assert suite.benchmarks[1].config == {"y": 20}
    assert suite.benchmarks[0].weight == 0.7


def test_parse_suite_file_with_path_object(tmp_path):
    suite_data = {
        "name": "suite_path",
        "benchmarks": [{"name": "benchA", "config": {"foo": "bar"}}],
    }
    suite_file = tmp_path / "suite_path.json"
    suite_file.write_text(json.dumps(suite_data))
    suite = parse_suite_file(Path(suite_file))
    assert suite.name == "suite_path"
    assert suite.benchmarks[0].name == "benchA"


def test_parse_suite_file_invalid_json(tmp_path):
    suite_file = tmp_path / "invalid.json"
    suite_file.write_text("{invalid json}")
    with pytest.raises(json.JSONDecodeError):
        parse_suite_file(suite_file)


def test_parse_suite_file_invalid_schema(tmp_path):
    suite_data = {"invalid": "data"}
    suite_file = tmp_path / "invalid_schema.json"
    suite_file.write_text(json.dumps(suite_data))
    with pytest.raises(Exception):
        parse_suite_file(suite_file)


def test_suite_weights_must_sum_to_one(tmp_path):
    suite_data = {
        "name": "bad_weights",
        "benchmarks": [
            {"name": "bench1", "config": {"x": 1}, "weight": 0.5},
            {"name": "bench2", "config": {"y": 2}, "weight": 0.6},
        ],
    }
    suite_file = tmp_path / "bad_weights.json"
    suite_file.write_text(json.dumps(suite_data))
    with pytest.raises(ValueError):
        parse_suite_file(suite_file)
