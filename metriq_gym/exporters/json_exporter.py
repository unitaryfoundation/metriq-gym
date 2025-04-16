import json
from metriq_gym.exporters.base_exporter import BaseExporter


class JsonExporter(BaseExporter):
    def export(self, filename=None) -> None:
        if not filename:
            filename = f"{self.metriq_gym_job.job_id}.json"
        with open(filename, "w") as json_file:
            json.dump(self._as_dict(), json_file, indent=4)
