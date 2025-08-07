from typing import Any
from metriq_gym.exporters.base_exporter import BaseExporter


class DictExporter(BaseExporter):
    def export(self) -> dict[str, Any]:
        return self.as_dict()
