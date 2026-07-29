import pytest
import json
from pathlib import Path
from metriq_gym.suite_parser import BenchmarkEntry, Suite, parse_suite_file


def test_benchmark_entry_model():
    entry = BenchmarkEntry(name="test_benchmark", config={"param": 1})
    assert entry.name == "test_benchmark"
    assert entry.component is None
    assert entry.component_name == "test_benchmark"
    assert entry.config == {"param": 1}


def test_suite_model():
    entry1 = BenchmarkEntry(name="b1_small", component="b1", config={"a": 1})
    entry2 = BenchmarkEntry(name="b1_large", component="b1", config={"b": 2})
    suite = Suite(
        name="suite1",
        version="1.0",
        description="A test suite",
        source="https://example.com",
        full_suite_warning="Expensive suite",
        benchmarks=[entry1, entry2],
    )
    assert suite.name == "suite1"
    assert suite.version == "1.0"
    assert suite.description == "A test suite"
    assert suite.source == "https://example.com"
    assert suite.full_suite_warning == "Expensive suite"
    assert len(suite.benchmarks) == 2
    assert suite.benchmarks[0].name == "b1_small"
    assert suite.benchmarks[1].config == {"b": 2}
    assert suite.component_names == ["b1"]


def test_suite_select_components_groups_entries_and_preserves_suite_order():
    suite = Suite(
        name="suite",
        benchmarks=[
            BenchmarkEntry(name="qft_4", component="qft", config={}),
            BenchmarkEntry(name="wit", component="wit", config={}),
            BenchmarkEntry(name="qft_8", component="qft", config={}),
        ],
    )

    selected = suite.select_components(["WIT", "QFT"])

    assert [entry.name for entry in selected] == ["qft_4", "wit", "qft_8"]


def test_suite_select_components_rejects_unknown_component():
    suite = Suite(
        name="suite",
        benchmarks=[BenchmarkEntry(name="wit", config={})],
    )

    with pytest.raises(ValueError, match="Unknown component.*Available components: wit"):
        suite.select_components(["qft"])


def test_suite_select_components_rejects_duplicate_selection():
    suite = Suite(
        name="suite",
        benchmarks=[BenchmarkEntry(name="qft_4", component="qft", config={})],
    )

    with pytest.raises(ValueError, match="Duplicate component selection"):
        suite.select_components(["qft", "QFT"])


def test_suite_keeps_legacy_duplicate_names_selectable():
    suite = Suite(
        name="suite",
        benchmarks=[
            BenchmarkEntry(name="QFT", config={"size": 4}),
            BenchmarkEntry(name="qft", config={"size": 8}),
        ],
    )

    assert [entry.config["size"] for entry in suite.select_components(["qft"])] == [4, 8]


def test_parse_suite_file(tmp_path):
    suite_data = {
        "name": "suite_test",
        "benchmarks": [
            {"name": "bench1", "config": {"x": 10}},
            {"name": "bench2", "config": {"y": 20}},
        ],
    }
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps(suite_data))
    suite = parse_suite_file(suite_file)
    assert suite.name == "suite_test"
    assert len(suite.benchmarks) == 2
    assert suite.benchmarks[0].name == "bench1"
    assert suite.benchmarks[1].config == {"y": 20}


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


def test_parse_bundled_suite_by_name():
    suite = parse_suite_file("uf_frugal_3")

    assert suite.name == "verified_uf_suite"


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
