"""Shared helpers for interacting with OriginQ's QCloud SDK."""

import os
from threading import Lock
from typing import Any

from pyqpanda3 import qcloud as qcloud_module

from ._constants import ENV_KEYS

_SERVICE_CACHE: dict[str, Any] = {}
_SERVICE_LOCK = Lock()


def resolve_api_key(explicit_key: str | None = None) -> str:
    """Resolve the OriginQ API key from explicit input or supported environment variables."""
    if explicit_key:
        return explicit_key
    for name in ENV_KEYS:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "OriginQ API key not configured. Set one of ORIGIN_API_KEY, ORIGINQ_API_KEY, or WUKONG_API_KEY."
    )


def get_service(explicit_key: str | None = None):
    """Return a cached QCloudService instance configured with the resolved API key."""
    key = resolve_api_key(explicit_key)
    with _SERVICE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None:
            service = qcloud_module.QCloudService(api_key=key)
            _SERVICE_CACHE[key] = service
    return service


def get_qcloud_job(job_id: str, *, api_key: str | None = None):
    """Instantiate a QCloudJob after ensuring the service is configured."""
    get_service(explicit_key=api_key)
    return qcloud_module.QCloudJob(job_id)


def get_qcloud_options():
    """Factory helper for QCloudOptions to keep imports centralized."""
    return qcloud_module.QCloudOptions()
