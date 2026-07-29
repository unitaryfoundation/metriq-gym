import json
from collections import Counter
from pathlib import Path

from metriq_gym.schema_validator import load_schema, validate_and_create_model
from metriq_gym.suite_parser import parse_suite_file


SUITE_PATH = Path(__file__).parents[2] / "metriq_gym" / "suites" / "metriq_score_1_0.json"
QFT_SWEEP_PATH = Path(__file__).parents[2] / "metriq_gym" / "suites" / "qft_sweep.json"
EXPECTED_CONFIGS: dict[str, dict] = {
    "bseq": {
        "benchmark_name": "BSEQ",
        "shots": 1000,
    },
    "eplg_100q_chain": {
        "benchmark_name": "EPLG",
        "num_qubits_in_chain": 100,
        "lengths": [2, 4, 8, 16, 30, 50, 70, 100, 150, 200, 300, 500],
        "num_samples": 10,
        "shots": 1000,
        "seed": 12345,
        "two_qubit_gate": "cz",
        "one_qubit_basis_gates": ["rz", "rx", "x"],
        "decompose_clifford_ops": False,
        "constrain_rb_offset_b": False,
    },
    "clops_100q_twirled_session": {
        "benchmark_name": "CLOPS",
        "num_qubits": 100,
        "num_layers": 100,
        "num_circuits": 1000,
        "shots": 100,
        "seed": 12345,
        "two_qubit_gate": "cz",
        "mode": "twirled",
        "use_session": True,
    },
    "wit_7q": {
        "benchmark_name": "WIT",
        "num_qubits": 7,
        "shots": 8192,
    },
}

for width, layers in [(8, 64), (16, 32), (24, 16), (32, 8), (64, 4), (128, 2)]:
    EXPECTED_CONFIGS[f"mirror_circuits_{width}q_{layers}l"] = {
        "benchmark_name": "Mirror Circuits",
        "width": width,
        "num_layers": layers,
        "two_qubit_gate_prob": 0.5,
        "two_qubit_gate_name": "CNOT",
        "shots": 1000,
        "num_circuits": 10,
    }

for width in [10, 20, 30, 50]:
    EXPECTED_CONFIGS[f"qml_kernel_{width}q"] = {
        "benchmark_name": "QML Kernel",
        "num_qubits": width,
        "shots": 1000,
    }

for width in [10, 20, 50, 100]:
    EXPECTED_CONFIGS[f"lr_qaoa_1d_{width}q"] = {
        "benchmark_name": "Linear Ramp QAOA",
        "graph_type": "1D",
        "num_qubits": width,
        "qaoa_layers": [10],
        "delta_beta": 0.3,
        "delta_gamma": 0.6,
        "shots": 1000,
        "trials": 10,
        "num_random_trials": 25,
        "confidence_level": 0.999,
        "seed": 123,
    }

for width in [4, 8, 12, 20]:
    EXPECTED_CONFIGS[f"qft_{width}q"] = {
        "benchmark_name": "Quantum Fourier Transform",
        "shots": 1000,
        "num_qubits": width,
        "max_circuits": 3,
        "method": 1,
        "use_midcircuit_measurement": False,
    }

EXPECTED_COMPONENTS = {
    "bseq": "bseq",
    "eplg_100q_chain": "eplg",
    "clops_100q_twirled_session": "clops",
    "wit_7q": "wit",
    **{
        f"mirror_circuits_{width}q_{layers}l": "mirror-circuits"
        for width, layers in [(8, 64), (16, 32), (24, 16), (32, 8), (64, 4), (128, 2)]
    },
    **{f"qml_kernel_{width}q": "qml-kernel" for width in [10, 20, 30, 50]},
    **{f"lr_qaoa_1d_{width}q": "lr-qaoa" for width in [10, 20, 50, 100]},
    **{f"qft_{width}q": "qft" for width in [4, 8, 12, 20]},
}


def test_metriq_score_1_0_definition_is_exact():
    raw_suite = json.loads(SUITE_PATH.read_text())
    raw_entries = {entry["name"]: entry for entry in raw_suite["benchmarks"]}

    assert raw_suite["name"] == "metriq_score_1_0"
    assert raw_suite["version"] == "1.0"
    assert raw_suite["source"] == "https://arxiv.org/abs/2603.08680v1"
    assert "22" in raw_suite["full_suite_warning"]
    assert len(raw_entries) == 22
    assert set(raw_entries) == set(EXPECTED_CONFIGS)

    for name, expected_config in EXPECTED_CONFIGS.items():
        assert raw_entries[name]["component"] == EXPECTED_COMPONENTS[name]
        assert raw_entries[name]["config"] == expected_config


def test_metriq_score_1_0_components_and_configs_are_valid():
    suite = parse_suite_file(SUITE_PATH)

    assert suite.component_names == [
        "bseq",
        "eplg",
        "mirror-circuits",
        "clops",
        "qml-kernel",
        "wit",
        "lr-qaoa",
        "qft",
    ]
    assert Counter(entry.component_name for entry in suite.benchmarks) == {
        "bseq": 1,
        "eplg": 1,
        "mirror-circuits": 6,
        "clops": 1,
        "qml-kernel": 4,
        "wit": 1,
        "lr-qaoa": 4,
        "qft": 4,
    }

    for entry in suite.benchmarks:
        schema = load_schema(entry.config["benchmark_name"])
        assert set(entry.config) <= set(schema["properties"])
        validate_and_create_model(entry.config)


def test_legacy_qft_sweep_is_not_repurposed_as_the_score_suite():
    suite = parse_suite_file(QFT_SWEEP_PATH)

    assert [entry.config["num_qubits"] for entry in suite.benchmarks] == [10, 20, 30, 50]
