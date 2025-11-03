from pprint import pprint

from metriq_gym.exporters.base_exporter import BaseExporter


class CliExporter(BaseExporter):
    def export(self) -> None:
        # Print metadata first (without the nested results block)
        record = self.as_dict()
        record.pop("results", None)
        pprint(record)

        # Print results using the model's computed properties
        values = self.result.values
        uncertainties = self.result.uncertainties
        if values:
            print("\nResults:")
            for key in sorted(values):
                value = values[key]
                # Preserve explicit 0.0; treat None/empty as missing
                uncertainty = uncertainties.get(key)
                if uncertainty is None or uncertainty == "":
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value} ± {uncertainty}")
