"""MkDocs hooks for generating documentation."""

import subprocess
import sys
from pathlib import Path


def on_pre_build(config):
    """Generate benchmark documentation before building."""
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "generate_benchmark_docs.py"
    output = project_root / "docs" / "docs" / "benchmarks" / "overview.md"
    
    if script.exists():
        print(f"Generating benchmark docs from {script}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            if result.returncode == 0:
                output.write_text(result.stdout)
                print(f"Generated {output}")
            else:
                print(f"Warning: Script failed: {result.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not generate benchmark docs: {e}", file=sys.stderr)
