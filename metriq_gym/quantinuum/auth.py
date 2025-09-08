from __future__ import annotations

import os
from typing import Any


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

    # Single supported method (for now): email + password
    email = os.getenv("QUANTINUUM_EMAIL")
    password = os.getenv("QUANTINUUM_PASSWORD")

    if not email or not password:
        raise RuntimeError(
            "Quantinuum credentials not found. Set QUANTINUUM_EMAIL and QUANTINUUM_PASSWORD in your .env."
        )

    api = QuantinuumAPI()
    # Populate env vars expected by pytket-quantinuum variants
    os.environ["PYTKET_QUANTINUUM_USERNAME"] = email
    os.environ["PYTKET_QUANTINUUM_PASSWORD"] = password
    os.environ["HQS_EMAIL"] = email
    os.environ["HQS_PASSWORD"] = password
    os.environ["QUANTINUUM_EMAIL"] = email
    os.environ["QUANTINUUM_PASSWORD"] = password

    if hasattr(api, "login"):
        try:
            api.login()  # type: ignore[attr-defined]
        except TypeError:
            if hasattr(api, "set_user_credentials"):
                api.set_user_credentials(email, password)  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    "Unable to authenticate with QUANTINUUM_EMAIL/QUANTINUUM_PASSWORD. Update pytket-quantinuum."
                )
    elif hasattr(api, "set_user_credentials"):
        api.set_user_credentials(email, password)  # type: ignore[attr-defined]
    else:
        raise RuntimeError(
            "Unable to authenticate with QUANTINUUM_EMAIL/QUANTINUUM_PASSWORD. Update pytket-quantinuum."
        )

    return api
