"""Check an installed distribution, without pytest or access to the source checkout.

Copy this script to a temporary directory and run it with a fresh virtualenv's
Python after installing a wheel or source archive. Do not use an editable install.
"""

import argparse
from datetime import datetime, timedelta
from importlib import import_module, metadata, resources
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_installed_module(name: str) -> None:
    module = import_module(name)
    if module.__file__ is None:
        raise RuntimeError(f"No installed file found for {name}")
    path = Path(module.__file__).resolve()
    require(
        path.is_relative_to(Path(sys.prefix).resolve()),
        f"{name} was imported from outside the test virtualenv: {path}",
    )


def check_package(expected_version: str | None) -> tuple[str, Path]:
    require(sys.prefix != sys.base_prefix, "Run this check inside a fresh virtualenv")
    check_installed_module("metriq_gym")

    from metriq_gym import __version__

    distribution = metadata.distribution("metriq-gym")
    require(
        distribution.version == __version__,
        f"Metadata version {distribution.version} differs from runtime version {__version__}",
    )
    if expected_version is not None:
        require(
            __version__ == expected_version,
            f"Expected version {expected_version}, installed {__version__}",
        )
    require(
        any(
            entry.group == "console_scripts"
            and entry.name == "mgym"
            and entry.value == "metriq_gym.run:main"
            for entry in distribution.entry_points
        ),
        "The distribution does not register the mgym console script",
    )
    cli = Path(sysconfig.get_path("scripts")) / ("mgym.exe" if os.name == "nt" else "mgym")
    require(cli.is_file(), f"The installed mgym console script is missing: {cli}")
    require(
        cli.resolve().is_relative_to(Path(sys.prefix).resolve()),
        f"The mgym console script is outside the test virtualenv: {cli}",
    )

    from metriq_gym.benchmarks.qedc_benchmarks import QEDC_BENCHMARK_IMPORTS

    # These submodule packages are imported dynamically by benchmark handlers.
    for name in [
        "_common.metrics",
        *QEDC_BENCHMARK_IMPORTS.values(),
        "qiskit_device_benchmarking.utilities.gate_map",
    ]:
        check_installed_module(name)

    from metriq_gym.constants import SCHEMA_MAPPING
    from metriq_gym.schema_validator import load_schema
    from metriq_gym.suite_parser import parse_suite_file

    package = resources.files("metriq_gym")
    for benchmark, filename in SCHEMA_MAPPING.items():
        schema = json.loads(package.joinpath("schemas", filename).read_text(encoding="utf-8"))
        require(bool(schema.get("properties")), f"Empty schema for {benchmark}")
        require(load_schema(benchmark) == schema, f"Cannot load packaged schema for {benchmark}")

    suites = package.joinpath("suites")
    suite_names = {entry.name for entry in suites.iterdir() if entry.name.endswith(".json")}
    require(
        {"lr_qaoa.json", "metriq_score_1_0.json"} <= suite_names,
        f"Required bundled suites are missing: found {sorted(suite_names)}",
    )
    for name in sorted(suite_names):
        require(bool(parse_suite_file(name).benchmarks), f"Bundled suite {name} is empty")
    dashboard = package.joinpath("dashboard", "index.html").read_text(encoding="utf-8")
    require("<html" in dashboard.lower(), "The packaged dashboard HTML is missing or empty")
    print(
        f"Package {__version__}: installed imports, entry point, and resources passed", flush=True
    )
    return __version__, cli


def run_cli(cli: Path, directory: Path, *args: str) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        MGYM_LOCAL_DB_DIR=str(directory / "jobs"),
        MGYM_LOCAL_SIMULATOR_CACHE_DIR=str(directory / "simulator-cache"),
        NO_COLOR="1",
        PYTHONIOENCODING="utf-8",
    )
    print(f"Running mgym {' '.join(args)}", flush=True)
    result = subprocess.run(
        [str(cli), *args],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    # Include both streams in CI logs, including failures with a zero exit code.
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    result.check_returncode()
    return result.stdout


def validate_record(record: dict, version: str, suite_id: str | None) -> None:
    require(record["app_version"] == version, "Result records the wrong package version")
    require(record["job_type"] == "QML Kernel", "Result records the wrong benchmark")
    require(record["suite_id"] == suite_id, "Result records the wrong suite ID")
    require(record["platform"]["provider"] == "local", "Result records the wrong provider")
    require(record["platform"]["device"] == "aer_simulator", "Result records the wrong device")
    timestamp = datetime.fromisoformat(record["timestamp"])
    require(timestamp.utcoffset() == timedelta(0), "Result timestamp is not in UTC")
    require(not record.get("outcome"), "Benchmark failed instead of producing results")
    score = record["results"]["accuracy_score"]
    for name in ("value", "uncertainty"):
        value = score[name]
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            f"Invalid accuracy score {name}: {value}",
        )
    require(math.isclose(score["value"], 1.0), f"Unexpected simulator accuracy: {score}")


def check_simulator(cli: Path, version: str, directory: Path) -> None:
    run_cli(cli, directory, "--help")
    config = {"benchmark_name": "QML Kernel", "num_qubits": 4, "shots": 10}
    job_config = directory / "job.json"
    job_config.write_text(json.dumps(config), encoding="utf-8")
    run_cli(
        cli, directory, "job", "dispatch", str(job_config), "-p", "local", "-d", "aer_simulator"
    )
    job_output = directory / "job-results.json"
    run_cli(cli, directory, "job", "poll", "latest", "--json", str(job_output))
    require(job_output.is_file(), "Single-job polling did not produce a JSON file")
    job_record = json.loads(job_output.read_text(encoding="utf-8"))
    require(isinstance(job_record, dict), "Single-job polling did not produce a JSON object")
    validate_record(job_record, version, None)

    suite_config = directory / "suite.json"
    suite_config.write_text(
        json.dumps(
            {
                "name": "distribution_smoke",
                "benchmarks": [
                    {"name": f"qml_kernel_{qubits}q", "config": config | {"num_qubits": qubits}}
                    for qubits in (4, 3)
                ],
            }
        ),
        encoding="utf-8",
    )
    dispatched = run_cli(
        cli, directory, "suite", "dispatch", str(suite_config), "-p", "local", "-d", "aer_simulator"
    )
    match = re.search(r"metriq-gym Suite ID (\S+)", dispatched)
    if match is None:
        raise RuntimeError("Suite dispatch did not report a suite ID")
    suite_id = match.group(1)
    suite_output = directory / "suite-results.json"
    run_cli(cli, directory, "suite", "poll", suite_id, "--json", str(suite_output))
    require(suite_output.is_file(), "Suite polling did not produce a JSON file")
    records = json.loads(suite_output.read_text(encoding="utf-8"))
    require(isinstance(records, list) and len(records) == 2, "Expected two suite result records")
    for record in records:
        validate_record(record, version, suite_id)
    require(
        [record["params"]["num_qubits"] for record in records] == [4, 3],
        "Suite results lost benchmark parameters or dispatch order",
    )
    print("Local simulator: single-job and two-job suite JSON polling passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", help="Require this exact distribution version")
    args = parser.parse_args()
    version, cli = check_package(args.expected_version)
    with tempfile.TemporaryDirectory(prefix="metriq-gym-smoke-") as directory:
        check_simulator(cli, version, Path(directory))


if __name__ == "__main__":
    main()
