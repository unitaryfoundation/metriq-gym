from pprint import pprint

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.exporters.base_exporter import BaseExporter


class CliExporter(BaseExporter):
    def export(self) -> None:
        # Print metadata (keep results block separate for clarity below)
        record = self.as_dict()
        pprint(record)

        # Surface the full result object, formatting uncertainties inline when available
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
            value = payload[key]
            # If payload already carries value/uncertainty (e.g., BenchmarkScore), format inline
            if isinstance(value, dict) and "value" in value and "uncertainty" in value:
                if value["uncertainty"] is None or value["uncertainty"] == "":
                    print(f"  {key}: {value['value']}")
                else:
                    print(f"  {key}: {value['value']} ± {value['uncertainty']}")
            # Otherwise, if we have a separate uncertainty entry, format it
            elif key in result_uncertainties and result_uncertainties[key] is not None:
                print(f"  {key}: {value} ± {result_uncertainties[key]}")
            else:
                print(f"  {key}: {value}")
