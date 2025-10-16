from pprint import pprint
from typing import Any, Mapping

from metriq_gym.exporters.base_exporter import BaseExporter


class CliExporter(BaseExporter):
    def export(self) -> None:
        record = self.as_dict()
        results_section = record.pop("results", {})
        pprint(record)

        values: Mapping[str, Any]
        uncertainties: Mapping[str, Any]
        if isinstance(results_section, Mapping):
            values = (
                results_section.get("values", {})
                if isinstance(results_section.get("values", {}), Mapping)
                else {}
            )
            uncertainties = (
                results_section.get("uncertainties", {})
                if isinstance(results_section.get("uncertainties", {}), Mapping)
                else {}
            )
        else:
            values = {}
            uncertainties = {}

        if values:
            print("\nResults:")
            for key in sorted(values):
                value = values[key]
                uncertainty = uncertainties.get(key, "")
                if uncertainty in (None, "", 0):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value} ± {uncertainty}")
