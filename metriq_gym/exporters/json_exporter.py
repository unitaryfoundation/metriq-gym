import json
from metriq_gym.exporters.base_exporter import BaseExporter


class JsonExporter(BaseExporter):
    def export(self, filename: str | None = None) -> None:
        if not filename:
            filename = f"{self.metriq_gym_job.id}.json"
        with open(filename, "w") as json_file:
            json.dump(self.as_dict(), json_file, indent=4)
        print(f"Results exported to {filename}")
