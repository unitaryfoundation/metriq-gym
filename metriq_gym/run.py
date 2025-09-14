import argparse
from dataclasses import asdict
from datetime import datetime
import os
import sys
import logging
import uuid
from dotenv import load_dotenv

from qbraid import QbraidError
from qbraid.runtime import (
    get_providers,
    GateModelResultData,
    JobStatus,
    QuantumDevice,
    QuantumProvider,
    load_job,
    load_provider,
)
from tabulate import tabulate

from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.registry import (
    BENCHMARK_DATA_CLASSES,
    BENCHMARK_HANDLERS,
    BENCHMARK_RESULT_CLASSES,
    get_available_benchmarks,
)
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.cli import list_jobs, parse_arguments, prompt_for_job
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.exporters.cli_exporter import CliExporter
from metriq_gym.exporters.json_exporter import JsonExporter
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.qplatform.job import job_status
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.constants import JobType
from metriq_gym.suite_parser import parse_suite_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("metriq_gym")

available_benchmarks = get_available_benchmarks()

COMMON_SUITE_KEYS = ["provider", "device", "timestamp", "app_version"]


def setup_device(provider_name: str, backend_name: str) -> QuantumDevice:
    """
    Setup a QBraid device with id backend_name from specified provider.

    Args:
        provider_name: a metriq-gym supported provider name.
        backend_name: the id of a device supported by the provider.
    Raises:
        QBraidSetupError: If no device matching the name is found in the provider.
    """
    try:
        provider: QuantumProvider = load_provider(provider_name)
    except QbraidError:
        # Best-effort fallback for custom providers that may conflict with other plugins
        if provider_name in {"qnexus", "quantinuum_nexus"}:
            try:
                from metriq_gym.quantinuum.provider import QuantinuumProvider

                provider = QuantinuumProvider()
            except Exception:
                logger.error(f"No provider matching the name '{provider_name}' found.")
                logger.error(f"Providers available: {get_providers()}")
                raise QBraidSetupError("Provider not found")
        else:
            logger.error(f"No provider matching the name '{provider_name}' found.")
            logger.error(f"Providers available: {get_providers()}")
            raise QBraidSetupError("Provider not found")

    try:
        device = provider.get_device(backend_name)
    except QbraidError:
        logger.error(
            f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        )
        logger.error(f"Devices available: {[device.id for device in provider.get_devices()]}")
        raise QBraidSetupError("Device not found")
    return device


def setup_benchmark(args, params, job_type: JobType) -> Benchmark:
    return BENCHMARK_HANDLERS[job_type](args, params)


def setup_job_data_class(job_type: JobType) -> type[BenchmarkData]:
    return BENCHMARK_DATA_CLASSES[job_type]


def setup_benchmark_result_class(job_type: JobType) -> type[BenchmarkResult]:
    return BENCHMARK_RESULT_CLASSES[job_type]


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch a single benchmark configuration to a quantum device.

    Args:
        args: CLI arguments with benchmark config, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print(f"Starting dispatch on {args.provider}:{args.device}...")

    try:
        device: QuantumDevice = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    config_file = args.config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    # Load and validate the benchmark configuration
    params = load_and_validate(config_file)

    # Validate that the benchmark exists
    if params.benchmark_name not in available_benchmarks:
        print(
            f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
        )
        return

    job_type = JobType(params.benchmark_name)

    print(f"Dispatching {params.benchmark_name}...")

    handler: Benchmark = setup_benchmark(args, params, job_type)
    job_data: BenchmarkData = handler.dispatch_handler(device)

    job_id = job_manager.add_job(
        MetriqGymJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            params=params.model_dump(exclude_none=True),
            data=asdict(job_data),
            provider_name=args.provider,
            device_name=args.device,
            dispatch_time=datetime.now(),
        )
    )

    print(f"✓ {params.benchmark_name} dispatched with metriq-gym Job ID: {job_id}")


def dispatch_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch multiple benchmark configurations to a quantum device.

    Enables comprehensive device characterization by running the same benchmark
    type with different parameters or multiple benchmark types in sequence.

    Args:
        args: CLI arguments with benchmark configs, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print(f"Starting suite dispatch on {args.provider}:{args.device}...")

    try:
        device: QuantumDevice = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    config_file = args.suite_config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    # Load and validate the benchmark configuration
    suite = parse_suite_file(config_file)
    if not suite.benchmarks:
        print(f"✗ {config_file}: No benchmarks found in the suite")
        return

    results = []
    successful_jobs = []

    suite_id = str(uuid.uuid4())
    for benchmark_entry in suite.benchmarks:
        try:
            params = validate_and_create_model(benchmark_entry.config)

            # Validate that the benchmark exists
            if params.benchmark_name not in available_benchmarks:
                results.append(
                    f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
                )
                continue

            job_type = JobType(params.benchmark_name)

            print(
                f"Dispatching {benchmark_entry.name} ({params.benchmark_name}) from {suite.name}..."
            )

            handler: Benchmark = setup_benchmark(args, params, job_type)
            job_data: BenchmarkData = handler.dispatch_handler(device)

            job_id = job_manager.add_job(
                MetriqGymJob(
                    suite_id=suite_id,
                    id=str(uuid.uuid4()),
                    job_type=job_type,
                    params=params.model_dump(exclude_none=True),
                    data=asdict(job_data),
                    provider_name=args.provider,
                    device_name=args.device,
                    dispatch_time=datetime.now(),
                )
            )

            results.append(
                f"✓ {benchmark_entry.name} ({params.benchmark_name}) from {suite.name} dispatched with metriq-gym Job ID: {job_id}"
            )
            successful_jobs.append(job_id)

        except Exception as e:
            error_details = f"{type(e).__name__}: {str(e)}"
            results.append(f"✗ {benchmark_entry.name} from {suite.name} failed: {error_details}")

    print("\nSummary:")
    for result in results:
        print(f"  {result}")

    print(f"\nDispatch complete for suite {suite.name} with metriq-gym Suite ID {suite_id}")
    print(f"\nSuccessfully dispatched {len(successful_jobs)}/{len(suite.benchmarks)} benchmarks.")
    if successful_jobs:
        print("Use 'mgym suite poll' or 'mgym job poll' to check suite/job status.")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if not metriq_job:
        return
    print("Polling job...")
    result = fetch_result(metriq_job, args, job_manager)
    if result is None:
        print(f"Job {metriq_job.id} is not yet completed or has no results.")
        return
    export_job_result(args, metriq_job, result)


def poll_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to poll suite results.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    results: list[BenchmarkResult] = []
    for metriq_job in jobs:
        result = fetch_result(metriq_job, args, job_manager)
        if result is None:
            print(f"Job {metriq_job.id} is not yet completed or has no results.")
            return
        results.append(result)
    export_suite_results(args, jobs, results)


def print_selected(d, selected_keys):
    for k in selected_keys:
        if k in d:
            print(f"{k}: {d[k]}")


def export_suite_results(args, jobs: list[MetriqGymJob], results: list[BenchmarkResult]) -> None:
    if not jobs:
        return

    records = []
    for job, result in zip(jobs, results):
        records.append(DictExporter(job, result).export() | {"params": job.params})

    if hasattr(args, "json"):
        raise NotImplementedError("JSON export of suite results is not implemented yet.")
    else:
        print("\n--- Suite Metadata ---")
        print_selected(records[0], COMMON_SUITE_KEYS)
        print("\n--- Suite Results ---")
        print(tabulate_job_results(records))


def tabulate_job_results(records, sep=" – "):
    rows = []
    metric_keys = set()
    for record in records:
        metric_keys.update(record["results"].keys())
    metric_keys = sorted(metric_keys)

    headers = ["Job Type", "Parameters"] + metric_keys

    for record in records:
        name = record.get("job_type")
        params = record.get("params", {})
        if isinstance(params, dict):
            params_str = ", ".join(
                f"{k}={v}" for k, v in sorted(params.items()) if k != "benchmark_name"
            )
        else:
            params_str = str(params)
        row = [name, params_str]
        for metric in metric_keys:
            row.append(record["results"].get(metric, ""))
        rows.append(row)

    return tabulate(rows, headers=headers, floatfmt=".4g")


def export_job_result(
    args: argparse.Namespace, metriq_job: MetriqGymJob, result: BenchmarkResult
) -> None:
    if hasattr(args, "json"):
        JsonExporter(metriq_job, result).export(args.json)
    else:
        CliExporter(metriq_job, result).export()


def fetch_result(
    metriq_job: MetriqGymJob, args: argparse.Namespace, job_manager: JobManager
) -> BenchmarkResult | None:
    job_type: JobType = JobType(metriq_job.job_type)
    job_result_type = setup_benchmark_result_class(job_type)
    if metriq_job.result_data is not None:
        return job_result_type.model_validate(metriq_job.result_data)

    job_data: BenchmarkData = setup_job_data_class(job_type)(**metriq_job.data)
    handler = setup_benchmark(args, validate_and_create_model(metriq_job.params), job_type)
    # Build provider-specific QuantumJob instances
    if metriq_job.provider_name in {"qnexus", "quantinuum_nexus"}:
        from metriq_gym.quantinuum.job import QuantinuumJob, normalize_job_id

        # Normalize any legacy saved job ids (e.g., full JobRef repr strings)
        normalized_ids = [normalize_job_id(jid) for jid in job_data.provider_job_ids]
        if normalized_ids != job_data.provider_job_ids:
            # Persist fix so future polls work without re-dispatch
            job_data.provider_job_ids = normalized_ids  # type: ignore[attr-defined]
            metriq_job.data["provider_job_ids"] = normalized_ids
            try:
                job_manager.update_job(metriq_job)
            except Exception:
                pass
        quantum_jobs = [QuantinuumJob(job_id) for job_id in normalized_ids]
    else:
        quantum_jobs = [
            (load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data)))
            for job_id in job_data.provider_job_ids
        ]

    # Fast path: if status says completed for all, collect results
    statuses = [task.status() for task in quantum_jobs]
    if all(st == JobStatus.COMPLETED for st in statuses):
        result_data: list[GateModelResultData] = [task.result().data for task in quantum_jobs]
        result: BenchmarkResult = handler.poll_handler(job_data, result_data, quantum_jobs)
        metriq_job.result_data = result.model_dump()
        job_manager.update_job(metriq_job)
        return result

    # Fallback for providers that don't expose a clean status: if tasks already have results, treat as completed
    result_data: list[GateModelResultData] = []
    all_ready = True
    for task in quantum_jobs:
        # Some provider jobs (like qnexus) expose a quick readiness check
        ready = False
        if hasattr(task, "is_ready") and callable(getattr(task, "is_ready")):
            try:
                ready = bool(task.is_ready())  # type: ignore[attr-defined]
            except Exception:
                ready = False
        if ready or job_status(task).status == JobStatus.COMPLETED:
            try:
                result_data.append(task.result().data)
            except Exception:
                all_ready = False
                break
        else:
            all_ready = False
            break

    if all_ready and len(result_data) == len(quantum_jobs):
        result: BenchmarkResult = handler.poll_handler(job_data, result_data, quantum_jobs)
        metriq_job.result_data = result.model_dump()
        job_manager.update_job(metriq_job)
        return result

    # Provider-specific fallback: for qnexus, attempt to fetch results even if status appears unknown
    if metriq_job.provider_name in {"qnexus", "quantinuum_nexus"}:
        forced_data: list[GateModelResultData] = []
        failures: list[str] = []
        for task in quantum_jobs:
            try:
                forced_data.append(task.result().data)
            except Exception as e:
                failures.append(f"{task.id}: {type(e).__name__}: {e}")
        if forced_data and len(forced_data) == len(quantum_jobs):
            result: BenchmarkResult = handler.poll_handler(job_data, forced_data, quantum_jobs)
            metriq_job.result_data = result.model_dump()
            job_manager.update_job(metriq_job)
            return result
        if os.getenv("MGYM_QNEXUS_DEBUG") and failures:
            print("[mgym qnexus] forced result fetch failures:")
            for line in failures:
                print("  ", line)

    print("Job is not yet completed. Current status of tasks:")
    for task in quantum_jobs:
        info = job_status(task)
        msg = f"- {task.id}: {info.status.value}"
        if info.queue_position is not None:
            msg += f" (position {info.queue_position})"
        print(msg)
    print("Please try again later.")
    return None


def view_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if metriq_job:
        print(metriq_job)


def view_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to view suite jobs.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    print(f"Jobs for suite ID {args.suite_id}:")
    list_jobs(jobs, show_index=False, show_suite_id=False)


def delete_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if metriq_job:
        try:
            job_manager.delete_job(metriq_job.id)
            print(f"Job {metriq_job.id} deleted successfully.")
        except ValueError:
            print(f"Error: Job {metriq_job.id} could not be deleted. It may not exist.")
        except Exception as e:
            print(f"An unexpected error occurred while deleting the job: {e}")
    else:
        print("No job selected for deletion.")


def delete_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to delete suite jobs.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    for job in jobs:
        try:
            job_manager.delete_job(job.id)
            print(f"Job {job.id} deleted successfully.")
        except Exception as e:
            print(f"Failed to delete job {job.id}: {e}")
    print(f"All jobs for suite ID {args.suite_id} deleted successfully.")


def debug_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Provider-specific debug to resolve and fetch qNexus job details.

    Prints low-level information useful to adapt to SDK variations.
    """
    metriq_job = prompt_for_job(args, job_manager)
    if not metriq_job:
        return

    print("Debugging job...")
    print(f"Metriq Job ID: {metriq_job.id}")
    print(f"Provider: {metriq_job.provider_name}")
    print(f"Device: {metriq_job.device_name}")
    print(f"Provider Job IDs: {metriq_job.data.get('provider_job_ids')}")

    if metriq_job.provider_name not in {"qnexus", "quantinuum_nexus"}:
        print("This debug tool currently supports only qnexus.")
        return

    try:
        import qnexus as qnx  # type: ignore
    except Exception as e:
        print("Failed to import qnexus:", type(e).__name__, str(e))
        return

    from metriq_gym.quantinuum.job import _get_job_obj, _build_execute_proxy
    from metriq_gym.quantinuum.utils import ensure_login

    ensure_login()
    proj_name = os.getenv("QNEXUS_PROJECT_NAME")
    print("QNEXUS_PROJECT_NAME:", proj_name)
    proj_ref = None
    if proj_name:
        try:
            proj_ref = qnx.projects.get_or_create(name=proj_name)
            print("Project resolved:", proj_ref)
        except Exception as e:
            print("Project get_or_create failed:", type(e).__name__, str(e))

    for pid in metriq_job.data.get("provider_job_ids", []):
        print("\n--- Provider Job ID:", pid)
        # Try get
        for variant in (
            ("get(id)", lambda: qnx.jobs.get(pid)),
            ("get(uuid)", lambda: qnx.jobs.get(__import__('uuid').UUID(str(pid)))),
            ("get(id, project)", lambda: qnx.jobs.get(pid, project=proj_ref)),
            ("get(job_id=..., project)", lambda: qnx.jobs.get(job_id=pid, project=proj_ref)),
            ("get(id=..., project)", lambda: qnx.jobs.get(id=pid, project=proj_ref)),
        ):
            name, func = variant
            try:
                ref = func()
                print(f"qnx.jobs.{name} ->", type(ref), getattr(ref, "job_type", None), getattr(ref, "type", None))
                try:
                    print("ref.id:", getattr(ref, "id", None))
                except Exception:
                    pass
                # Try status and df
                try:
                    st = getattr(ref, "status", None)
                    if callable(st):
                        print("ref.status():", st())
                except Exception as e:
                    print("ref.status() error:", type(e).__name__, str(e))
                try:
                    df = getattr(ref, "df", lambda: None)()
                    if df is not None:
                        print("ref.df columns:", list(getattr(df, "columns", [])))
                except Exception as e:
                    print("ref.df() error:", type(e).__name__, str(e))
                # Try results
                try:
                    res = qnx.jobs.results(ref)
                    print("jobs.results(ref) count:", len(res) if res is not None else None)
                except Exception as e:
                    print("jobs.results(ref) error:", type(e).__name__, str(e))
                break
            except Exception as e:
                print(f"qnx.jobs.{name} error:", type(e).__name__, str(e))

        # Try listing
        try:
            listing = None
            # Attempt multiple list-like locations
            for path in (
                "jobs",
                "client.jobs",
            ):
                try:
                    mod = __import__(f"qnexus.{path}", fromlist=["list"])
                    if hasattr(mod, "list"):
                        listing = getattr(mod, "list")(project=proj_ref) if proj_ref else getattr(mod, "list")()
                        break
                except Exception:
                    continue
            if listing is None:
                print("jobs.list not available in this SDK")
            else:
                print("jobs.list type:", type(listing))
                try:
                    items = getattr(listing, "items", lambda: [])()
                    print("jobs.list items count:", len(items))
                    match = [r for r in items if str(getattr(r, "id", "")).lower() == str(pid).lower()]
                    print("items match count:", len(match))
                    if match:
                        r = match[0]
                        print("match type:", type(r), getattr(r, "job_type", None), getattr(r, "type", None))
                except Exception as e:
                    print("jobs.list items error:", type(e).__name__, str(e))
                try:
                    res = getattr(listing, "results", lambda: [])()
                    print("jobs.list results count:", len(res))
                except Exception as e:
                    print("jobs.list results error:", type(e).__name__, str(e))
                try:
                    df = getattr(listing, "df", lambda: None)()
                    print("jobs.list df columns:", list(getattr(df, "columns", [])) if df is not None else None)
                    if df is not None:
                        m = df[df[df.columns[0]].astype(str).str.lower() == str(pid).lower()] if len(df.columns) else None
                        print("df uuid match count:", 0 if m is None else len(m))
                except Exception as e:
                    print("jobs.list df error:", type(e).__name__, str(e))
        except Exception as e:
            print("jobs.list error:", type(e).__name__, str(e))

        # Try proxy
        try:
            proxy = _build_execute_proxy(pid)
            print("proxy type/job_type:", type(proxy), getattr(proxy, "job_type", None), getattr(proxy, "type", None))
            try:
                res = qnx.jobs.results(proxy)
                print("jobs.results(proxy) count:", len(res) if res is not None else None)
            except Exception as e:
                print("jobs.results(proxy) error:", type(e).__name__, str(e))
        except Exception as e:
            print("build proxy error:", type(e).__name__, str(e))


def main() -> int:
    load_dotenv()
    args = parse_arguments()
    job_manager = JobManager()

    RESOURCE_ACTION_TABLE = {
        "suite": {
            "dispatch": dispatch_suite,
            "poll": poll_suite,
            "delete": delete_suite,
        },
        "job": {
            "dispatch": dispatch_job,
            "poll": poll_job,
            "view": view_job,
            "delete": delete_job,
            "debug": debug_job,
        },
    }

    resource_table = RESOURCE_ACTION_TABLE.get(args.resource)
    if resource_table:
        action_handler = resource_table.get(args.action)
        if action_handler:
            action_handler(args, job_manager)
            return 0
    logging.error("Invalid command. Run with --help for usage information.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
