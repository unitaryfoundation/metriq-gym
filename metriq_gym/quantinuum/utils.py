from __future__ import annotations

import os


def _is_authenticated() -> bool:  # pragma: no cover - network/auth
    try:
        import qnexus as qnx  # type: ignore

        # Try root-level and namespaced checks
        if hasattr(qnx, "is_authenticated") and callable(qnx.is_authenticated):
            return bool(qnx.is_authenticated())
        if hasattr(qnx, "auth") and hasattr(qnx.auth, "is_authenticated"):
            return bool(qnx.auth.is_authenticated())  # type: ignore[attr-defined]
    except Exception:
        pass
    return False


def ensure_login() -> None:  # pragma: no cover - network/auth
    """Log into qNexus using env vars if not already authenticated.

    Supports multiple SDK variants:
    - qnx.login(api_key=...)
    - qnx.login(username=..., password=...)
    - qnx.login_with_api_key(...)
    - qnx.login_with_credentials(username=..., password=...) (or interactive)
    """
    import qnexus as qnx  # type: ignore

    if _is_authenticated():
        return

    api_key = os.getenv("QUANTINUUM_API_KEY")
    username = os.getenv("QUANTINUUM_USERNAME")
    password = os.getenv("QUANTINUUM_PASSWORD")

    # Prefer explicit credentials if provided
    # 1) Generic login(api_key=...)
    if api_key and hasattr(qnx, "login"):
        try:
            qnx.login(api_key=api_key)
            return
        except Exception:
            pass

    # 2) Generic login(username=..., password=...)
    if username and password and hasattr(qnx, "login"):
        try:
            qnx.login(username=username, password=password)
            return
        except Exception:
            pass

    # 3) Dedicated helpers if present
    if api_key and hasattr(qnx, "login_with_api_key"):
        try:
            qnx.login_with_api_key(api_key)
            return
        except Exception:
            pass

    if hasattr(qnx, "login_with_credentials"):
        try:
            if username and password:
                qnx.login_with_credentials(username=username, password=password)
            else:
                qnx.login_with_credentials()  # may prompt interactively
            return
        except Exception:
            pass

    # If we reach here, the SDK didn’t accept any path; leave auth to caller
    return
