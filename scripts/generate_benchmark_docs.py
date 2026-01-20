#!/usr/bin/env python3
"""Generate benchmark documentation from docstrings and JSON schemas.

This script creates a single source of truth for benchmark documentation by:
1. Extracting module docstrings from benchmark Python files
2. Parsing JSON schemas for configuration parameters
3. Generating markdown output for the docs

Usage:
    python scripts/generate_benchmark_docs.py > docs/docs/benchmarks/overview.md
    
Or use as a mkdocs hook (see docs/mkdocs.yml).
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metriq_gym.constants import JobType, SCHEMA_MAPPING

# Map JobType to Python module names
MODULE_MAPPING = {
    JobType.BSEQ: "metriq_gym.benchmarks.bseq",
    JobType.CLOPS: "metriq_gym.benchmarks.clops",
    JobType.EPLG: "metriq_gym.benchmarks.eplg",
    JobType.QML_KERNEL: "metriq_gym.benchmarks.qml_kernel",
    JobType.QUANTUM_VOLUME: "metriq_gym.benchmarks.quantum_volume",
    JobType.MIRROR_CIRCUITS: "metriq_gym.benchmarks.mirror_circuits",
    JobType.WIT: "metriq_gym.benchmarks.wit",
    JobType.LR_QAOA: "metriq_gym.benchmarks.lr_qaoa",
    # QEDC benchmarks are in a single module
    JobType.BERNSTEIN_VAZIRANI: "metriq_gym.benchmarks.qedc_benchmarks",
    JobType.PHASE_ESTIMATION: "metriq_gym.benchmarks.qedc_benchmarks",
    JobType.HIDDEN_SHIFT: "metriq_gym.benchmarks.qedc_benchmarks",
    JobType.QUANTUM_FOURIER_TRANSFORM: "metriq_gym.benchmarks.qedc_benchmarks",
}

# Benchmarks to include in the main documentation (skip QEDC variants for now)
DOCUMENTED_BENCHMARKS = [
    JobType.QUANTUM_VOLUME,
    JobType.CLOPS,
    JobType.MIRROR_CIRCUITS,
    JobType.EPLG,
    JobType.BSEQ,
    JobType.WIT,
    JobType.LR_QAOA,
    JobType.QML_KERNEL,
]


def load_schema(job_type: JobType) -> dict[str, Any]:
    """Load JSON schema for a benchmark."""
    schema_file = SCHEMA_MAPPING.get(job_type)
    if not schema_file:
        return {}
    
    schema_path = PROJECT_ROOT / "metriq_gym" / "schemas" / schema_file
    if not schema_path.exists():
        return {}
    
    with open(schema_path) as f:
        return json.load(f)


def load_example(job_type: JobType) -> dict[str, Any] | None:
    """Load example configuration for a benchmark."""
    # Convert job type to example filename
    schema_file = SCHEMA_MAPPING.get(job_type, "")
    example_name = schema_file.replace(".schema.json", ".example.json")
    example_path = PROJECT_ROOT / "metriq_gym" / "schemas" / "examples" / example_name
    
    if not example_path.exists():
        return None
    
    with open(example_path) as f:
        return json.load(f)


def get_module_docstring(job_type: JobType) -> str:
    """Get the module-level docstring for a benchmark.
    
    Reads the docstring directly from the source file to avoid import errors
    from missing dependencies.
    """
    module_name = MODULE_MAPPING.get(job_type)
    if not module_name:
        return ""
    
    # Convert module name to file path
    module_path = module_name.replace(".", "/") + ".py"
    file_path = PROJECT_ROOT / module_path
    
    if not file_path.exists():
        return ""
    
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree) or ""
    except Exception as e:
        print(f"Warning: Could not read docstring from {file_path}: {e}", file=sys.stderr)
        return ""


def parse_docstring(docstring: str) -> dict[str, str]:
    """Parse a structured docstring into sections.
    
    Expected format:
        One-line summary or multi-line summary.
        
        Summary:
            ...
        
        Result interpretation:
            ...
        
        References:
            ...
    """
    sections = {}
    current_section = "summary"
    current_content = []
    
    # Known section headers we care about
    known_sections = {
        "summary": "summary",
        "result interpretation": "result_interpretation",
        "references": "references",
        "connectivity graph": "connectivity_graph",
    }
    
    for line in docstring.split("\n"):
        stripped = line.strip()
        
        # Check if this line is a section header
        # Must be at the start of the line (not indented) or only slightly indented
        # and end with a colon
        if stripped.endswith(":") and not stripped.startswith("-"):
            # Check if it's a known section header
            potential_header = stripped[:-1].lower()
            if potential_header in known_sections:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = known_sections[potential_header]
                current_content = []
                continue
        
        current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def format_parameter_table(schema: dict[str, Any]) -> str:
    """Generate a markdown table from schema properties."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    if not properties:
        return ""
    
    lines = [
        "| Parameter | Type | Required | Default | Description |",
        "|-----------|------|----------|---------|-------------|",
    ]
    
    for name, prop in properties.items():
        if name == "benchmark_name":
            continue  # Skip the benchmark_name field
        
        prop_type = prop.get("type", "any")
        is_required = "Yes" if name in required else "No"
        default = prop.get("default", "—")
        if default != "—":
            default = f"`{default}`"
        description = prop.get("description", "")
        
        # Handle constraints
        constraints = []
        if "minimum" in prop:
            constraints.append(f"min: {prop['minimum']}")
        if "maximum" in prop:
            constraints.append(f"max: {prop['maximum']}")
        if constraints:
            description += f" ({', '.join(constraints)})"
        
        lines.append(f"| `{name}` | {prop_type} | {is_required} | {default} | {description} |")
    
    return "\n".join(lines)


def format_benchmark_section(job_type: JobType) -> str:
    """Generate markdown section for a single benchmark."""
    schema = load_schema(job_type)
    docstring = get_module_docstring(job_type)
    example = load_example(job_type)
    
    title = schema.get("title", str(job_type))
    sections = parse_docstring(docstring)
    
    lines = [f"### {title}", ""]
    
    # Summary from docstring or schema description
    summary_text = None
    if "summary" in sections:
        summary = sections["summary"]
        # Clean up indentation
        summary_lines = [line.strip() for line in summary.split("\n") if line.strip()]
        summary_text = " ".join(summary_lines)
    
    # If docstring is just a one-liner, use it as summary
    if not summary_text and docstring:
        first_line = docstring.strip().split("\n")[0].strip()
        if first_line and not first_line.endswith(":"):
            summary_text = first_line
    
    # Fall back to schema description
    if not summary_text and schema.get("description"):
        summary_text = schema["description"]
    
    if summary_text:
        lines.append(summary_text)
        lines.append("")
    
    # Result interpretation from docstring
    if "result_interpretation" in sections:
        lines.append("**Result interpretation:**")
        lines.append("")
        result_text = sections["result_interpretation"]
        # Clean up and format
        for line in result_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append(stripped)
            elif stripped:
                lines.append(stripped)
        lines.append("")
    
    # Parameters from schema
    param_table = format_parameter_table(schema)
    if param_table:
        lines.append("**Parameters:**")
        lines.append("")
        lines.append(param_table)
        lines.append("")
    
    # Example configuration
    if example:
        lines.append("**Example configuration:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(example, indent=2))
        lines.append("```")
        lines.append("")
    
    # References from docstring
    if "references" in sections:
        lines.append("**References:**")
        lines.append("")
        for line in sections["references"].split("\n"):
            stripped = line.strip()
            if stripped.startswith("- "):
                lines.append(stripped)
            elif stripped:
                lines.append(f"- {stripped}")
        lines.append("")
    
    return "\n".join(lines)


def generate_benchmarks_page() -> str:
    """Generate the complete benchmarks overview page."""
    lines = [
        "# Benchmarks",
        "",
        "Metriq-Gym provides a comprehensive suite of quantum benchmarks to characterize and compare quantum hardware performance.",
        "",
        "<!-- This file is auto-generated by scripts/generate_benchmark_docs.py -->",
        "<!-- Do not edit manually. Instead, update docstrings and schemas. -->",
        "",
        "## Running Benchmarks",
        "",
        "### Dispatch a Benchmark",
        "",
        "```bash",
        "mgym job dispatch <config.json> --provider <provider> --device <device>",
        "```",
        "",
        "### Poll for Results",
        "",
        "```bash",
        "mgym job poll <JOB_ID>",
        "```",
        "",
        "## Configuration",
        "",
        "All benchmarks use JSON configuration files validated against JSON schemas.",
        "",
        "- **Example configurations**: [`metriq_gym/schemas/examples/`](https://github.com/unitaryfoundation/metriq-gym/tree/main/metriq_gym/schemas/examples)",
        "- **JSON schemas**: [`metriq_gym/schemas/`](https://github.com/unitaryfoundation/metriq-gym/tree/main/metriq_gym/schemas)",
        "",
        "## Available Benchmarks",
        "",
    ]
    
    for job_type in DOCUMENTED_BENCHMARKS:
        lines.append(format_benchmark_section(job_type))
        lines.append("---")
        lines.append("")
    
    # Remove last separator
    lines = lines[:-2]
    
    lines.extend([
        "",
        "## Additional Benchmarks",
        "",
        "The following benchmarks from the [QEDC benchmark suite](https://github.com/SRI-International/QC-App-Oriented-Benchmarks) are also available:",
        "",
        "- Bernstein-Vazirani",
        "- Phase Estimation", 
        "- Hidden Shift",
        "- Quantum Fourier Transform",
        "",
        "See their schema files in `metriq_gym/schemas/` for configuration options.",
        "",
        "## Adding Custom Benchmarks",
        "",
        "See [Adding New Benchmarks](../development/adding-benchmarks.md) to contribute new benchmarks.",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_benchmarks_page())
