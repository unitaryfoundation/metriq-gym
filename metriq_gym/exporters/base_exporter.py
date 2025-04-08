from abc import ABC, abstractmethod


class BaseExporter(ABC):
    @abstractmethod
    def submit(self, *args, **kwargs) -> None:
        pass
