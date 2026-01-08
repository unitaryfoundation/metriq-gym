import json
import re
import subprocess
import pytest
from pathlib import Path
from metriq_gym.schema_validator import load_schema

# Define paths relative to this test file
PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "metriq_gym" / "schemas" / "examples"


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", str(tmp_path))


def get_example_files():
    """Return all .json files in the examples directory."""
    # Exclude LR QAOA native layout which won't work with the fully connected
    # AER simulator device. Other LR QAOA examples should sufficiently exercise
    # that benchmark.
    excluded_files = {"lr_qaoa_native_layout.example.json"}
    res = [f for f in EXAMPLES_DIR.glob("*.json") if f.name not in excluded_files]
    return res


def get_min_value(prop_schema: dict):
    """Extract the robust minimum value for a specific property schema."""
    # 1. Handle Enum (e.g., WIT num_qubits: [6, 7] -> 6)
    if "enum" in prop_schema and prop_schema["enum"]:
        # Return the smallest valid option (e.g., 6 qubits instead of 7)
        return min(prop_schema["enum"])

    # 2. Handle Numeric Minimums
    if "minimum" in prop_schema:
        return prop_schema["minimum"]

    # 3. Handle Fixed Constants
    if "const" in prop_schema:
        return prop_schema["const"]

    return None


def shrink_config_using_schema(config: dict) -> dict:
    """
    Dynamically reduces config values to their schema-defined minimums.
    """
    benchmark_name = config.get("benchmark_name")
    if not benchmark_name:
        return config

    try:
        # Load the official schema for this benchmark
        schema = load_schema(benchmark_name)
    except (ValueError, FileNotFoundError):
        # Fail gracefully if schema is missing; the dispatch will likely fail later
        return config

    properties = schema.get("properties", {})

    for key, current_value in config.items():
        if key not in properties:
            continue

        prop_schema = properties[key]

        # A. Shrink simple Numbers/Integers
        schema_min = get_min_value(prop_schema)
        if schema_min is not None and isinstance(current_value, (int, float)):
            # Only update if the current example value is larger than the minimum
            if current_value > schema_min:
                config[key] = schema_min

        # B. Shrink Arrays (e.g., qaoa_layers, random_seeds)
        if prop_schema.get("type") == "array" and isinstance(current_value, list):
            min_items = prop_schema.get("minItems", 1)
            # 1. Truncate list to minimum length
            if len(current_value) > min_items:
                config[key] = current_value[:min_items]

            # 2. Shrink items inside the list (e.g. if list contains layer depths)
            items_schema = prop_schema.get("items", {})
            item_min = get_min_value(items_schema)
            if item_min is not None:
                # Set all remaining items to the allowed minimum
                config[key] = [
                    item_min if (isinstance(x, (int, float)) and x > item_min) else x
                    for x in config[key]
                ]

    # Special Case: Range-based benchmarks (QED-C)
    # If we shrunk max_qubits to its min (e.g. 2), ensure it doesn't conflict with min_qubits
    if "min_qubits" in config and "max_qubits" in config:
        # Collapse the range to the smallest single valid size
        config["max_qubits"] = config["min_qubits"]

    return config


@pytest.mark.e2e
@pytest.mark.parametrize("schema_path", get_example_files(), ids=lambda p: p.name)
def test_benchmark_schema_compliance_e2e(schema_path, tmp_path, monkeypatch):
    """
    Validates that every example file matches its schema AND executes successfully
    on the local simulator when parameters are minimized.
    """
    # 1. Load Original Config
    with open(schema_path) as f:
        original_config = json.load(f)

    # 2. Dynamic Shrinking
    # This uses the specific schema for *this* benchmark to find safe lower bounds
    shrunk_config = shrink_config_using_schema(original_config)

    # 3. Write Config to Temp File
    temp_config_path = tmp_path / f"shrunk_{schema_path.name}"
    with open(temp_config_path, "w") as f:
        json.dump(shrunk_config, f)

    # 5. Dispatch (This triggers the Schema Validator internally)
    # If parameters were renamed in schema but not code, this step explodes.
    # Uses fake_jakarta which has 7 qubits needed for WIT, but is smaller/restricited couplings
    # than the default AER simulator, which would be too large for BSEQ running locally.
    dispatch_cmd = subprocess.run(
        ["mgym", "job", "dispatch", str(temp_config_path), "-p", "local", "-d", "fake_jakarta"],
        capture_output=True,
        text=True,
    )

    if dispatch_cmd.returncode != 0:
        pytest.fail(
            f"Dispatch failed for {schema_path.name}:\n{dispatch_cmd.stdout}\n{dispatch_cmd.stderr}"
        )

    # 6. Extract Job ID
    # Output: "✓ <Name> dispatched with metriq-gym Job ID: <uuid>"
    # Use regex to extract just the UUID, since pyqrack may print warnings
    # to stdout after the job ID line (e.g., "No platforms found. Check OpenCL installation!")
    match = re.search(r"Job ID: ([0-9a-f-]+)", dispatch_cmd.stdout)
    if not match:
        pytest.fail(f"Could not extract Job ID. Stdout: {dispatch_cmd.stdout}")
    job_id = match.group(1)

    # 7. Poll (Verify execution)
    poll_cmd = subprocess.run(["mgym", "job", "poll", job_id], capture_output=True, text=True)

    if poll_cmd.returncode != 0:
        pytest.fail(f"Poll failed for {schema_path.name}:\n{poll_cmd.stderr}")

    assert "Result" in poll_cmd.stdout or "results" in poll_cmd.stdout.lower(), (
        f"No results detected in output for {schema_path.name}"
    )
