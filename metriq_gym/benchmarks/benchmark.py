import argparse

from pydantic import BaseModel
from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice


"""
Store intermediate data from pre-processing and dispatching
"""


@dataclass
class BenchmarkData:
    provider_job_ids: list[str]


"""
Store the final results of the benchmark
"""


@dataclass
class BenchmarkResult:
    pass


class Benchmark:
    def __init__(
        self,
        args: argparse.Namespace,
        params: BaseModel,
    ):
        self.args = args
        self.params: BaseModel = params

    def dispatch_handler(self, device: QuantumDevice) -> BenchmarkData:
        raise NotImplementedError

    def poll_handler(
        self,
        job_data: BenchmarkData,
        result_data: list[GateModelResultData],
    ) -> BenchmarkResult:
        raise NotImplementedError
