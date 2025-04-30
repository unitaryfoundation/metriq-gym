from pprint import pprint
from metriq_gym.exporters.base_exporter import BaseExporter


class CliExporter(BaseExporter):
    def export(self) -> None:
        pprint(self._as_dict())
