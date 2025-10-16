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
                uncertainty = uncertainties.get(key, 0)
                if not uncertainty:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value} ± {uncertainty}")
