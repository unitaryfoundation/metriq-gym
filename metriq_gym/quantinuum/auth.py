from __future__ import annotations

import os
from typing import Any


def _clean(val: str | None) -> str | None:
    if not val:
        return None
    v = val.strip()
    if not v or v.startswith("<") and v.endswith(">"):
        return None
    return v


def load_api() -> Any:
    """Create and authenticate a QuantinuumAPI instance.

    Supports username/password and (optionally) API key flows, handling
    minor version differences in pytket-quantinuum.
    """
    try:
        from pytket.extensions.quantinuum import QuantinuumAPI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pytket-quantinuum. Install with: poetry add pytket-quantinuum"
        ) from exc

    api_key = _clean(os.getenv("QUANTINUUM_API_KEY"))
    username = _clean(os.getenv("QUANTINUUM_USERNAME") or os.getenv("QUANTINUUM_EMAIL"))
    password = _clean(os.getenv("QUANTINUUM_PASSWORD"))

    # Prefer username/password for broad compatibility
    if username and password:
        api = QuantinuumAPI()
        # Populate all known env variable names used by various versions
        os.environ["PYTKET_QUANTINUUM_USERNAME"] = username
        os.environ["PYTKET_QUANTINUUM_PASSWORD"] = password
        os.environ["HQS_EMAIL"] = username
        os.environ["HQS_PASSWORD"] = password
        os.environ["QUANTINUUM_EMAIL"] = username
        os.environ["QUANTINUUM_PASSWORD"] = password
        # Try zero-arg login, fallback to explicit setter
        if hasattr(api, "login"):
            try:
                api.login()  # type: ignore[attr-defined]
            except TypeError:
                if hasattr(api, "set_user_credentials"):
                    api.set_user_credentials(username, password)  # type: ignore[attr-defined]
                else:
                    raise RuntimeError(
                        "Unable to authenticate with username/password. Please update pytket-quantinuum."
                    )
        elif hasattr(api, "set_user_credentials"):
            api.set_user_credentials(username, password)  # type: ignore[attr-defined]
        else:
            raise RuntimeError(
                "Unable to authenticate with username/password. Please update pytket-quantinuum."
            )
        return api

    if api_key:
        try:
            # Newer versions may support api_key in constructor
            return QuantinuumAPI(api_key=api_key)  # type: ignore[arg-type]
        except TypeError as exc:
            raise RuntimeError(
                "Your pytket-quantinuum version does not support API key constructor. "
                "Use QUANTINUUM_USERNAME/QUANTINUUM_PASSWORD instead or upgrade pytket-quantinuum."
            ) from exc

    raise RuntimeError(
        "Quantinuum credentials not found. Set QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD (recommended), "
        "or QUANTINUUM_API_KEY if your pytket-quantinuum supports it."
    )

