"""Benchmark schemas reference common.schema.json rather than restating properties.

Kept out of test_schema_validator.py, which patches JobType module-wide with an
autouse fixture; these tests need the real job types.
"""

import glob
import json

import pytest

from metriq_gym.constants import SCHEMA_MAPPING
from metriq_gym.schema_validator import (
    _resolve_refs,
    create_pydantic_model,
    load_schema,
)

COMMON_SCHEMA = "metriq_gym/schemas/common.schema.json"


def test_ref_is_materialized_for_the_model_builder():
    """create_pydantic_model reads field_schema["type"], so refs must be inlined."""
    shots = load_schema("QML Kernel")["properties"]["shots"]
    assert "$ref" not in shots
    assert shots["type"] == "integer"
    assert shots["default"] == 1000


def test_sibling_keywords_override_the_shared_definition():
    """EPLG defaults to 100 shots; everything else takes the shared 1000."""
    assert load_schema("EPLG")["properties"]["shots"]["default"] == 100
    assert load_schema("CLOPS")["properties"]["shots"]["default"] == 1000


def test_every_schema_still_builds_a_model():
    for job_type in SCHEMA_MAPPING:
        create_pydantic_model(load_schema(job_type.value))


def test_unknown_reference_is_rejected():
    with pytest.raises(ValueError, match="Unknown schema reference"):
        _resolve_refs({"x": {"$ref": "common.schema.json#/$defs/nope"}}, {})


def test_shared_properties_are_referenced_not_restated():
    """Guards the drift this removed: shots had seven wordings across twelve files."""
    common = json.load(open(COMMON_SCHEMA))["$defs"]
    offenders = []
    for path in sorted(glob.glob("metriq_gym/schemas/*.schema.json")):
        if path.endswith("common.schema.json"):
            continue
        for name, defn in json.load(open(path)).get("properties", {}).items():
            if name in common and "$ref" not in defn:
                offenders.append(f"{path}:{name}")
    assert not offenders, f"restated instead of referenced: {offenders}"
