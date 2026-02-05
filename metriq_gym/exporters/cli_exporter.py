from pprint import pprint
from typing import Any

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.exporters.base_exporter import BaseExporter


def _format_result_line(key: str, value: Any, uncertainties: dict) -> str:
    """Format a single result metric for display."""
    if isinstance(value, dict) and "value" in value and "uncertainty" in value:
        if value["uncertainty"] is None or value["uncertainty"] == "":
            return f"  {key}: {value['value']}"
        return f"  {key}: {value['value']} ± {value['uncertainty']}"
    elif key in uncertainties and uncertainties[key] is not None:
        return f"  {key}: {value} ± {uncertainties[key]}"
    return f"  {key}: {value}"


class CliExporter(BaseExporter):
    def export(self) -> None:
        # Print metadata (keep results block separate for clarity below)
        record = self.as_dict()
        pprint(record)

        # Check if QEM was applied
        result_data = self.metriq_gym_job.result_data or {}
        qem_applied = result_data.get("_qem_applied", False)
        raw_result = result_data.get("_raw_result")
        qem_config = result_data.get("_qem_config", [])

        # Surface the full result object, formatting uncertainties inline when available
        if qem_applied:
            techniques = ", ".join(c.get("technique", "?") for c in qem_config)
            print(f"\nResults (mitigated via {techniques}):")
        else:
            print("\nResults:")

        payload = self.result.model_dump()
        result_uncertainties = self.result.uncertainties or {}
        score_val = getattr(self.result, "score", None)
        if isinstance(score_val, BenchmarkScore):
            payload["score"] = score_val.model_dump()
            result_uncertainties = {**result_uncertainties, "score": score_val.uncertainty}

        for key in sorted(payload.keys()):
            if key == "uncertainties":
                continue
            print(_format_result_line(key, payload[key], result_uncertainties))

        # If QEM was applied, also print the raw (unmitigated) results for comparison
        if qem_applied and raw_result:
            print(f"\nRaw (unmitigated) results:")
            for key in sorted(raw_result.keys()):
                if key in ("uncertainties", "score"):
                    continue
                value = raw_result[key]
                print(_format_result_line(key, value, {}))
