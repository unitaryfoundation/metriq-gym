"""Schema loading and validation utilities for metriq-gym benchmark configurations."""

import json
import os
from typing import Any
from jsonschema import validate
from pydantic import BaseModel, create_model, Field
from importlib import resources

from metriq_gym.constants import JobType, SCHEMA_MAPPING

SCHEMA_DIR_NAME = "schemas"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCHEMA_DIR = os.path.join(CURRENT_DIR, SCHEMA_DIR_NAME)
BENCHMARK_NAME_KEY = "benchmark_name"


def _schema_resource_path(filename: str) -> str | None:
    """Return a filesystem path to a bundled schema JSON file.

    Tries importlib.resources (works for wheels / site-packages). Falls back to
    direct path inside the source tree if not found. Returns None if neither exists.
    """
    # Attempt to access as a package resource
    try:
        candidate = resources.files("metriq_gym").joinpath(SCHEMA_DIR_NAME, filename)
        if candidate.is_file():
            # Convert to string path so existing loaders keep working
            return str(candidate)
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    # Fallback: direct path relative to source (development / editable install)
    direct_path = os.path.join(DEFAULT_SCHEMA_DIR, filename)
    if os.path.isfile(direct_path):
        return direct_path
    return None


def load_json_file(file_path: str) -> dict:
    """Load and parse a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def load_schema(benchmark_name: str, schema_dir: str = DEFAULT_SCHEMA_DIR) -> dict:
    """Load a JSON schema based on the benchmark name.

    Uses package resources for installed distributions; falls back to local path.
    """
    schema_filename = SCHEMA_MAPPING.get(JobType(benchmark_name))
    if not schema_filename:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    # Prefer packaged resource; allow overriding via explicit schema_dir argument
    if schema_dir != DEFAULT_SCHEMA_DIR:
        candidate = os.path.join(schema_dir, schema_filename)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Schema file not found: {candidate}")
        return load_json_file(candidate)

    resource_path = _schema_resource_path(schema_filename)
    if resource_path is None:
        raise FileNotFoundError(
            f"Schema file '{schema_filename}' not found in package resources or '{DEFAULT_SCHEMA_DIR}'."
        )
    return load_json_file(resource_path)


def create_pydantic_model(schema: dict[str, Any]) -> Any:
    """Create a Pydantic model from a JSON schema."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    required_fields = set(schema.get("required", []))
    field_definitions = {}

    for field_name, field_schema in schema["properties"].items():
        field_type = type_mapping[field_schema["type"]]
        has_default = "default" in field_schema
        is_required = field_name in required_fields

        # Build Field parameters
        field_params = {}

        if has_default:
            field_params["default"] = field_schema["default"]
        elif not is_required:
            field_params["default"] = None
        else:
            field_params["default"] = ...  # Required field

        # Add constraints if they exist
        if "minimum" in field_schema:
            field_params["ge"] = field_schema["minimum"]
        if "maximum" in field_schema:
            field_params["le"] = field_schema["maximum"]
        if "pattern" in field_schema:
            field_params["pattern"] = field_schema["pattern"]
        if "minLength" in field_schema:
            field_params["min_length"] = field_schema["minLength"]
        if "maxLength" in field_schema:
            field_params["max_length"] = field_schema["maxLength"]

        # Create final type, avoiding variable reassignment
        final_type = field_type | None if not is_required and not has_default else field_type
        field_definitions[field_name] = (final_type, Field(**field_params))

    model = create_model(schema["title"], **field_definitions)
    model.model_rebuild()
    return model


def validate_and_create_model(
    params: dict[str, Any], schema_dir: str = DEFAULT_SCHEMA_DIR
) -> BaseModel:
    if params.get(BENCHMARK_NAME_KEY) is None:
        raise ValueError(f"Missing {BENCHMARK_NAME_KEY} key in input file.")
    schema = load_schema(params[BENCHMARK_NAME_KEY], schema_dir)
    validate(params, schema)

    model = create_pydantic_model(schema)
    return model(**params)


def load_and_validate(file_path: str, schema_dir: str = DEFAULT_SCHEMA_DIR) -> BaseModel:
    """
    Load parameters from a JSON file and validate them against the corresponding schema.

    Raises a ValidationError if validation fails.
    """
    params = load_json_file(file_path)
    return validate_and_create_model(params, schema_dir)
