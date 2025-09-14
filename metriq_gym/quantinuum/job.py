from __future__ import annotations

from typing import Any
import os
import sys
import re
import uuid

from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


def _get_job_ref(job_id: str):  # pragma: no cover - requires qnexus
    """Backward-compatible alias retained; prefer _get_job_obj in new code."""
    return _get_job_obj(job_id) or job_id


def _get_job_obj(job_id: str):  # pragma: no cover - requires qnexus
    """Attempt to always return a concrete JobRef object, or None if not found."""
    import qnexus as qnx  # type: ignore
    from .utils import ensure_login

    ensure_login()

    # Try project-scoped first, then global
    proj_name = os.getenv("QNEXUS_PROJECT_NAME")
    proj_ref = None
    if proj_name:
        try:
            proj_ref = qnx.projects.get_or_create(name=proj_name)
        except Exception:
            proj_ref = None
    # Try reconstructing from persisted metadata first (from dispatch time)
    try:
        from ._store import read_meta

        meta = read_meta(str(job_id))
        if meta is not None:
            # Prefer constructing a real ExecuteJobRef if available
            ExecuteJobRefCls = None
            JobTypeEnum = None
            for path in (
                "qnexus.client.models.job_refs",
                "qnexus.models.job_refs",
                "qnexus.client.models.jobs",
                "qnexus.models.jobs",
                "qnexus.models.job",
                "qnexus.models",
            ):
                try:
                    mod = __import__(path, fromlist=["ExecuteJobRef", "JobType"])  # type: ignore[attr-defined]
                    if hasattr(mod, "ExecuteJobRef") and ExecuteJobRefCls is None:
                        ExecuteJobRefCls = getattr(mod, "ExecuteJobRef")
                    if hasattr(mod, "JobType") and JobTypeEnum is None:
                        JobTypeEnum = getattr(mod, "JobType")
                except Exception:
                    continue
            if ExecuteJobRefCls is not None:
                try:
                    uid = uuid.UUID(str(meta.get("id", job_id)))
                except Exception:
                    uid = job_id
                jt = meta.get("job_type", "execute")
                if JobTypeEnum is not None:
                    try:
                        jt = JobTypeEnum(str(jt))  # type: ignore[call-arg]
                    except Exception:
                        jt = getattr(JobTypeEnum, "EXECUTE", None) or str(jt)
                try:
                    return ExecuteJobRefCls(id=uid, job_type=jt)  # type: ignore[call-arg]
                except Exception:
                    try:
                        import qnexus as qnx  # type: ignore

                        proj_name = os.getenv("QNEXUS_PROJECT_NAME")
                        proj_ref = qnx.projects.get_or_create(name=proj_name) if proj_name else None
                        return ExecuteJobRefCls(id=uid, job_type=jt, project=proj_ref)  # type: ignore[call-arg]
                    except Exception:
                        try:
                            return ExecuteJobRefCls(id=uid)  # type: ignore[call-arg]
                        except Exception:
                            pass
    except Exception:
        pass

    # Try with raw string and UUID object, with and without project
    candidates = [job_id]
    try:
        candidates.append(uuid.UUID(str(job_id)))
    except Exception:
        pass
    for cand in candidates:
        try:
            if proj_ref is not None:
                return qnx.jobs.get(cand, project=proj_ref)  # type: ignore[call-arg]
            return qnx.jobs.get(cand)
        except Exception:
            try:
                # keyword forms
                if proj_ref is not None:
                    return qnx.jobs.get(job_id=cand, project=proj_ref)  # type: ignore[call-arg]
                return qnx.jobs.get(job_id=cand)
            except Exception:
                try:
                    if proj_ref is not None:
                        return qnx.jobs.get(id=cand, project=proj_ref)  # type: ignore[call-arg]
                    return qnx.jobs.get(id=cand)
                except Exception:
                    pass
    try:
        listing = qnx.jobs.list(project=proj_ref) if proj_ref is not None else qnx.jobs.list()
        # Prefer iterating actual items if available
        for attr in ("items", "results", "__iter__"):
            if hasattr(listing, attr):
                try:
                    seq = listing.items if attr == "items" else (listing.results if attr == "results" else list(listing))
                except Exception:
                    seq = None
                if seq:
                    for ref in seq:
                        # Accept refs with .id, .job_id, .uuid
                        for key in ("id", "job_id", "uuid"):
                            if hasattr(ref, key):
                                val = getattr(ref, key)
                                if str(val).lower() == str(job_id).lower():
                                    return ref
        # Fallback: pandas df
        df = getattr(listing, "df", lambda: None)()
        if df is not None:
            for col in ["id", "job_id", "uuid", "jobId"]:
                if col in df.columns:
                    hits = df[df[col].astype(str).str.lower() == str(job_id).lower()]
                    if not hits.empty:
                        val = hits.iloc[0][col]
                        try:
                            return qnx.jobs.get(val, project=proj_ref) if proj_ref is not None else qnx.jobs.get(val)
                        except Exception:
                            continue
            # Last attempt: substring match in records
            try:
                for rec in df.to_dict("records"):
                    for v in rec.values():
                        if str(job_id).lower() in str(v).lower():
                            return qnx.jobs.get(v, project=proj_ref) if proj_ref is not None else qnx.jobs.get(v)
            except Exception:
                pass
    except Exception:
        pass
    return None


def normalize_job_id(raw_id: str) -> str:  # pragma: no cover - requires qnexus
    """Normalize various saved job-id strings to a canonical UUID string.

    Handles previous runs where the full JobRef repr was persisted instead of the UUID.
    Attempts, in order:
    - Extract UUID-like substring
    - Extract UUID from patterns like "UUID('<uuid>')"
    - Extract job name from annotations and resolve via qnexus list() within project
    - Resolve via _get_job_obj and return its concrete id
    Fallback to the original string if no better candidate is found.
    """
    try:
        import qnexus as qnx  # type: ignore
    except Exception:
        return raw_id

    # 1) Direct UUID match in the string
    m = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw_id)
    if m:
        return m.group(0)

    # 2) Pattern UUID('...')
    m = re.search(r"UUID\('([0-9a-fA-F-]{36})'\)", raw_id)
    if m:
        return m.group(1)

    # 3) Extract annotations name='...' and lookup by name
    m = re.search(r"name='([^']+)'", raw_id)
    if m:
        name = m.group(1)
        try:
            proj_name = os.getenv("QNEXUS_PROJECT_NAME")
            proj_ref = qnx.projects.get_or_create(name=proj_name) if proj_name else None
            df = (qnx.jobs.list(project=proj_ref).df() if proj_ref else qnx.jobs.list().df())
            if "name" in getattr(df, "columns", []):
                hits = df[df["name"] == name]
                if not hits.empty:
                    for col in ["id", "job_id", "uuid", "jobId"]:
                        if col in df.columns:
                            return str(hits.iloc[0][col])
        except Exception:
            pass

    # 4) Resolve via API
    obj = _get_job_obj(raw_id)
    if obj is not None:
        for attr in ("id", "job_id", "uuid", "uid"):
            if hasattr(obj, attr):
                try:
                    return str(getattr(obj, attr))
                except Exception:
                    continue

    return raw_id


def _build_execute_proxy(job_id: str | uuid.UUID):  # pragma: no cover - requires qnexus
    """Construct a minimal ExecuteJobRef-like object for APIs that expect a ref.

    Some qnexus versions accept any object with `.id` and `.job_type` attributes.
    """
    # Try to import real model classes for better compatibility
    JobTypeEnum = None
    ExecuteJobRefCls = None
    for path in (
        "qnexus.client.models.job_refs",
        "qnexus.client.models.jobs",
        "qnexus.models.job_refs",
        "qnexus.models.jobs",
        "qnexus.models.job",
        "qnexus.models",
    ):
        try:
            mod = __import__(path, fromlist=["JobType", "ExecuteJobRef"])  # type: ignore[attr-defined]
            if hasattr(mod, "JobType") and JobTypeEnum is None:
                JobTypeEnum = getattr(mod, "JobType")
            if hasattr(mod, "ExecuteJobRef") and ExecuteJobRefCls is None:
                ExecuteJobRefCls = getattr(mod, "ExecuteJobRef")
            break
        except Exception:
            continue

    # Try to instantiate a real ExecuteJobRef if available
    try:
        if ExecuteJobRefCls is not None:
            uid = uuid.UUID(str(job_id))
            # Determine job type enum or fallback
            if JobTypeEnum is not None:
                try:
                    jt = JobTypeEnum("execute")
                except Exception:
                    jt = getattr(JobTypeEnum, "EXECUTE", None) or "execute"
            else:
                jt = "execute"
            # Try several constructor signatures
            try:
                return ExecuteJobRefCls(id=uid, job_type=jt)  # type: ignore[call-arg]
            except Exception:
                pass
            try:
                # Some constructors may expect 'project' as well
                import qnexus as qnx  # type: ignore
                proj_name = os.getenv("QNEXUS_PROJECT_NAME")
                proj_ref = qnx.projects.get_or_create(name=proj_name) if proj_name else None
                return ExecuteJobRefCls(id=uid, job_type=jt, project=proj_ref)  # type: ignore[call-arg]
            except Exception:
                pass
            try:
                return ExecuteJobRefCls(id=uid)  # type: ignore[call-arg]
            except Exception:
                pass
    except Exception:
        pass

    class _Proxy:
        def __init__(self, _id):
            try:
                self.id = uuid.UUID(str(_id))
            except Exception:
                self.id = _id
            # qnexus often inspects either 'job_type' or 'type'
            if JobTypeEnum is not None:
                try:
                    # Prefer enum by value if constructor supports it
                    try:
                        self.job_type = JobTypeEnum("execute")  # type: ignore[call-arg]
                    except Exception:
                        self.job_type = getattr(JobTypeEnum, "EXECUTE", None) or "execute"
                except Exception:
                    self.job_type = "execute"
            else:
                self.job_type = "execute"
            # Hint the ref kind; many SDKs use 'ExecuteJobRef'
            try:
                self.type = "ExecuteJobRef"
            except Exception:
                pass

    return _Proxy(job_id)


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)

    def is_ready(self) -> bool:  # pragma: no cover - requires qnexus
        try:
            import qnexus as qnx  # type: ignore
        except Exception:
            return False
        job_ref = _get_job_ref(self.id)
        try:
            results = qnx.jobs.results(job_ref)
            return bool(results)
        except Exception:
            return False

    def result(self) -> Result:  # pragma: no cover - requires qnexus
        import qnexus as qnx  # type: ignore
        try:
            from qnexus.exceptions import AuthenticationError  # type: ignore
        except Exception:
            class AuthenticationError(Exception):
                pass

        from .utils import ensure_login

        job_ref = _get_job_obj(self.id)
        if job_ref is None:
            if os.getenv("MGYM_QNEXUS_DEBUG"):
                print(
                    f"[mgym qnexus] could not build JobRef for id {self.id}; using execute proxy",
                    file=sys.stderr,
                )
            job_ref = _build_execute_proxy(self.id)
        # First try to fetch results without blocking
        results = []
        def _fetch_results():
            try:
                # Try common call signatures
                try:
                    return qnx.jobs.results(job_ref)
                except Exception:
                    pass
                try:
                    return qnx.jobs.results(job_id=getattr(job_ref, "id", self.id))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return qnx.jobs.results(id=getattr(job_ref, "id", self.id))  # type: ignore[call-arg]
                except Exception:
                    pass
                # Final fallback to raw values
                try:
                    return qnx.jobs.results(uuid.UUID(str(self.id)))
                except Exception:
                    return qnx.jobs.results(self.id)
            except AuthenticationError:
                ensure_login()
                try:
                    return qnx.jobs.results(job_ref)
                except Exception:
                    pass
                try:
                    return qnx.jobs.results(job_id=getattr(job_ref, "id", self.id))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return qnx.jobs.results(id=getattr(job_ref, "id", self.id))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return qnx.jobs.results(uuid.UUID(str(self.id)))
                except Exception:
                    return qnx.jobs.results(self.id)
        try:
            results = _fetch_results() or []
        except Exception as e:
            if os.getenv("MGYM_QNEXUS_DEBUG"):
                print(f"[mgym qnexus] results initial fetch error: {type(e).__name__}: {e}", file=sys.stderr)
            # Try alternate helpers if available
            try:
                if hasattr(qnx.jobs, "get_outputs"):
                    results = qnx.jobs.get_outputs(job_ref) or []
            except Exception as e2:
                if os.getenv("MGYM_QNEXUS_DEBUG"):
                    print(f"[mgym qnexus] get_outputs error: {type(e2).__name__}: {e2}", file=sys.stderr)
        # If no results yet, wait for completion and retry
        if not results:
            try:
                qnx.jobs.wait_for(job_ref)
            except AuthenticationError:
                ensure_login()
                qnx.jobs.wait_for(job_ref)
            try:
                results = _fetch_results() or results
            except Exception:
                pass
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")
        # Download pytket BackendResult and convert to counts
        item0 = results[0]
        backend_result = None
        counts = None
        # Try the canonical method
        if hasattr(item0, "download_result"):
            backend_result = item0.download_result()
            if hasattr(backend_result, "get_counts"):
                counts = backend_result.get_counts()
        # Fallbacks if SDK differs
        if counts is None and hasattr(item0, "download_counts"):
            counts = item0.download_counts()
        if counts is None and hasattr(item0, "get_output"):
            try:
                output = item0.get_output()
                if hasattr(output, "get_counts"):
                    counts = output.get_counts()
            except Exception:
                pass
        if counts is None:
            raise RuntimeError("Result object does not provide counts")

        return Result(
            device_id=self.device.id if self.device else "quantinuum",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=counts),
        )

    def status(self) -> JobStatus:  # pragma: no cover - requires qnexus
        try:
            import qnexus as qnx  # type: ignore
            try:
                from qnexus.exceptions import AuthenticationError  # type: ignore
            except Exception:
                class AuthenticationError(Exception):
                    pass

            job_ref = _get_job_obj(self.id)
            if job_ref is None:
                if os.getenv("MGYM_QNEXUS_DEBUG"):
                    print(
                        f"[mgym qnexus] status: using execute proxy for id {self.id}",
                        file=sys.stderr,
                    )
                job_ref = _build_execute_proxy(self.id)
            # Many SDKs provide .status or a details endpoint; try both
            status = getattr(job_ref, "status", None)
            if callable(status):
                try:
                    status = status()
                except AuthenticationError:
                    from .utils import ensure_login

                    ensure_login()
                    status = status()
            if not status:
                try:
                    # pandas dataframe view
                    df = job_ref.df()
                    # Prefer explicit columns if present
                    col = None
                    for c in ["status", "state", "job_status", "job_state"]:
                        if hasattr(df, c):
                            col = c
                            break
                    if col is not None:
                        status = str(getattr(df, col).iloc[0])
                except Exception:
                    pass
            if not status:
                # Try direct attributes often present on refs
                for attr in ["last_status", "state", "status_text", "last_message"]:
                    if hasattr(job_ref, attr):
                        try:
                            s = getattr(job_ref, attr)
                            status = getattr(s, "value", s)
                            if status:
                                break
                        except Exception:
                            continue
            if not status:
                # If result is ready, consider completed
                try:
                    res = qnx.jobs.results(job_ref)
                except AuthenticationError:
                    from .utils import ensure_login

                    ensure_login()
                    res = qnx.jobs.results(job_ref)
                if res:
                    return JobStatus.COMPLETED
                return JobStatus.RUNNING

            status_str = str(status).upper()
            # Robust mapping across possible states
            if (
                "COMPLETE" in status_str
                or status_str == "COMPLETED"
                or "SUCCESS" in status_str
                or "SUCCEEDED" in status_str
                or "FINISHED" in status_str
                or "DONE" in status_str
                or "EXECUTED" in status_str
            ):
                return JobStatus.COMPLETED
            if "RUN" in status_str or status_str == "QUEUED" or status_str == "PENDING":
                return JobStatus.RUNNING
            if "CANCEL" in status_str:
                return JobStatus.CANCELLED
            if "FAIL" in status_str or "ERROR" in status_str:
                return JobStatus.ERROR
            return JobStatus.UNKNOWN
        except Exception as e:
            if os.getenv("MGYM_QNEXUS_DEBUG"):
                print(f"[mgym qnexus] status error: {type(e).__name__}: {e}", file=sys.stderr)
            return JobStatus.UNKNOWN

    def cancel(self) -> bool:  # pragma: no cover - requires qnexus
        try:
            job_ref = _get_job_ref(self.id)
            cancel = getattr(job_ref, "cancel", None)
            if callable(cancel):
                cancel()
                return True
        except Exception:
            pass
        return False
