import json
from pathlib import Path

import pytest

from metriq_gym.schema_validator import load_schema

EXAMPLES_DIR = Path(__file__).parents[2] / "metriq_gym" / "schemas" / "examples"


@pytest.mark.parametrize("example_file", sorted(EXAMPLES_DIR.glob("*.json")), ids=lambda p: p.name)
def test_example_keys_are_defined_in_schema(example_file):
    # Unknown keys are dropped silently during validation, so a misspelled
    # parameter would fall back to its default instead of failing.
    config = json.loads(example_file.read_text())
    schema = load_schema(config["benchmark_name"])
    assert set(config) - set(schema["properties"]) == set()
