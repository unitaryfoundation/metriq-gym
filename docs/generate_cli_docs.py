#!/usr/bin/env python3
"""Generate CLI documentation from Typer app.

Usage:
    python generate_cli_docs.py

This script generates markdown documentation for the CLI commands
by introspecting the Typer app structure.
"""

import inspect
import sys
from pathlib import Path
from typing import Annotated, get_args, get_origin

# Add parent directory to path to import metriq_gym
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from typer.models import OptionInfo

from metriq_gym.cli import job_app, suite_app


def extract_param_info(annotation, default):
    """Extract parameter info from Typer's Annotated type."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        typer_info = args[1] if len(args) > 1 else None

        # Get type name
        if hasattr(base_type, "__origin__"):
            # Handle Optional[str] -> str
            inner_args = get_args(base_type)
            if inner_args:
                type_name = inner_args[0].__name__ if hasattr(inner_args[0], "__name__") else "str"
            else:
                type_name = "str"
        else:
            type_name = base_type.__name__ if hasattr(base_type, "__name__") else "str"

        # Extract info from Typer's ArgumentInfo or OptionInfo
        help_text = ""
        param_decls = []
        envvar = None
        is_flag = False
        is_option = isinstance(typer_info, OptionInfo)

        if typer_info:
            help_text = getattr(typer_info, "help", "") or ""
            # Typer splits option names: first goes to 'default', rest to 'param_decls'
            typer_default = getattr(typer_info, "default", None)
            extra_decls = list(getattr(typer_info, "param_decls", []) or [])
            if is_option and isinstance(typer_default, str) and typer_default.startswith("-"):
                param_decls = [typer_default] + extra_decls
            else:
                param_decls = extra_decls
            envvar = getattr(typer_info, "envvar", None)
            is_flag = getattr(typer_info, "is_flag", False)

        return {
            "type": type_name.upper(),
            "help": help_text,
            "param_decls": param_decls,
            "envvar": envvar,
            "is_flag": is_flag,
            "is_option": is_option,
            "default": default,
            "required": default is inspect.Parameter.empty,
        }
    return {
        "type": "TEXT",
        "help": "",
        "param_decls": [],
        "envvar": None,
        "is_flag": False,
        "is_option": False,
        "default": default,
        "required": default is inspect.Parameter.empty,
    }


def generate_command_docs(cmd_info, parent_name: str) -> str:
    """Generate markdown documentation for a single command."""
    callback = cmd_info.callback
    name = cmd_info.name
    lines = []

    # Command header
    lines.append(f"## {name}\n")

    # Description from docstring
    docstring = callback.__doc__ or ""
    if docstring:
        # Get first paragraph of docstring
        first_para = docstring.strip().split("\n\n")[0].replace("\n", " ").strip()
        lines.append(f"{first_para}\n")

    # Get function signature
    sig = inspect.signature(callback)
    annotations = callback.__annotations__

    # Separate arguments and options
    arguments = []
    options = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue

        annotation = annotations.get(param_name, str)
        info = extract_param_info(annotation, param.default)
        info["name"] = param_name

        # Determine if it's an argument or option
        if info["is_option"]:
            options.append(info)
        else:
            arguments.append(info)

    # Build usage string
    usage_parts = ["mgym", parent_name, name]
    usage_args = []

    for arg in arguments:
        if arg["required"]:
            usage_args.append(f"<{arg['name']}>")
        else:
            usage_args.append(f"[{arg['name']}]")

    if options:
        usage_args.append("[OPTIONS]")

    usage = " ".join(usage_parts + usage_args)
    lines.append("```bash")
    lines.append(usage)
    lines.append("```\n")

    # Arguments table
    if arguments:
        lines.append("### Arguments\n")
        lines.append("| Argument | Type | Description | Required |")
        lines.append("|----------|------|-------------|----------|")
        for arg in arguments:
            name_upper = arg["name"].upper()
            required = "Yes" if arg["required"] else "No"
            lines.append(f"| `{name_upper}` | {arg['type']} | {arg['help']} | {required} |")
        lines.append("")

    # Options table
    if options:
        lines.append("### Options\n")
        lines.append("| Option | Type | Description | Default |")
        lines.append("|--------|------|-------------|---------|")
        for opt in options:
            # Use param_decls if available, otherwise derive from name
            if opt["param_decls"]:
                opt_str = ", ".join(opt["param_decls"])
            else:
                opt_str = f"--{opt['name'].replace('_', '-')}"
            help_text = opt["help"]
            if opt["envvar"]:
                help_text += f" (env: `{opt['envvar']}`)"
            default = opt["default"]
            if default is None or default is inspect.Parameter.empty:
                default = "None"
            type_name = "FLAG" if opt["is_flag"] else opt["type"]
            if opt["is_flag"]:
                default = "False"
            lines.append(f"| `{opt_str}` | {type_name} | {help_text} | `{default}` |")
        lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def generate_app_docs(subapp, app_name: str, title: str, description: str) -> str:
    """Generate documentation for all commands in a sub-app."""
    lines = [
        f"# {title}\n",
        f"{description}\n",
    ]

    for cmd_info in subapp.registered_commands:
        lines.append(generate_command_docs(cmd_info, app_name))

    return "\n".join(lines)


def generate_quick_reference(apps: list[tuple[str, typer.Typer]]) -> str:
    """Generate quick reference table from Typer apps."""
    lines = [
        "| Command | Description |",
        "|---------|-------------|",
    ]
    for app_name, app in apps:
        for cmd_info in app.registered_commands:
            callback = cmd_info.callback
            name = cmd_info.name
            # Get first line of docstring as description
            docstring = callback.__doc__ or ""
            description = docstring.strip().split("\n")[0].strip() if docstring else ""
            lines.append(f"| `mgym {app_name} {name}` | {description} |")
    return "\n".join(lines)


def main():
    """Generate CLI documentation files."""
    content_dir = Path(__file__).parent / "content" / "cli"
    content_dir.mkdir(parents=True, exist_ok=True)

    # Generate job commands docs
    job_docs = generate_app_docs(
        job_app,
        "job",
        "Job Commands",
        "Commands for dispatching, monitoring, and managing individual benchmark jobs.",
    )
    (content_dir / "job-commands.md").write_text(job_docs)
    print(f"Generated {content_dir / 'job-commands.md'}")

    # Generate suite commands docs
    suite_docs = generate_app_docs(
        suite_app,
        "suite",
        "Suite Commands",
        "Commands for dispatching, monitoring, and managing benchmark suites.",
    )
    (content_dir / "suite-commands.md").write_text(suite_docs)
    print(f"Generated {content_dir / 'suite-commands.md'}")

    # Generate quick reference table dynamically
    quick_ref = generate_quick_reference([("job", job_app), ("suite", suite_app)])

    # Generate overview
    overview = f"""# CLI Overview

Metriq-Gym provides a command-line interface (`mgym`) for dispatching, monitoring, and uploading quantum benchmark results.

## Installation

The CLI is installed automatically with metriq-gym:

```bash
pip install metriq-gym
```

## Command Structure

```
mgym <resource> <action> [arguments] [options]
```

Resources:
- `job` - Individual benchmark jobs
- `suite` - Collections of benchmark jobs

## Quick Reference

{quick_ref}

## Getting Help

```bash
# Main help
mgym --help

# Job commands help
mgym job --help

# Specific command help
mgym job dispatch --help
```
"""
    (content_dir / "overview.md").write_text(overview)
    print(f"Generated {content_dir / 'overview.md'}")


if __name__ == "__main__":
    main()
