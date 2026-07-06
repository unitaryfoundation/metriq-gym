"""Provider-specific hooks applied during device setup and job polling.

Centralises all provider workarounds so that ``run.py`` never needs
``if provider == "xyz"`` conditionals.  Each hook module registers itself
by adding entries to the dictionaries below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from qbraid.runtime import QuantumDevice

# {provider_name: callable(device) -> None}
_device_hooks: dict[str, Callable[["QuantumDevice"], None]] = {}

# {provider_name: callable() -> None}  (class-level / idempotent patches)
_poll_hooks: dict[str, Callable[[], None]] = {}

# {provider_name: callable(load_kwargs, params) -> None}  (mutates load_kwargs)
_load_kwargs_hooks: dict[str, Callable[[dict[str, Any], dict[str, Any]], None]] = {}


def _register_ionq() -> None:
    from metriq_gym.ionq.device import patch_ionq_device, patch_ionq_job

    _device_hooks["ionq"] = patch_ionq_device
    _poll_hooks["ionq"] = patch_ionq_job

    def _ionq_load_kwargs(load_kwargs: dict[str, Any], params: dict[str, Any]) -> None:
        if "shots" not in load_kwargs:
            load_kwargs["shots"] = params.get("shots")

    _load_kwargs_hooks["ionq"] = _ionq_load_kwargs


_register_ionq()


# ── Public API called by run.py ────────────────────────────────────────


def apply_device_hook(provider_name: str, device: "QuantumDevice") -> None:
    """Apply provider-specific patches to a device after construction."""
    hook = _device_hooks.get(provider_name)
    if hook is not None:
        hook(device)


def apply_poll_hooks(provider_name: str) -> None:
    """Apply provider-specific class-level patches before polling jobs."""
    hook = _poll_hooks.get(provider_name)
    if hook is not None:
        hook()


def apply_load_kwargs_hook(
    provider_name: str,
    load_kwargs: dict[str, Any],
    params: dict[str, Any],
) -> None:
    """Let a provider mutate load_kwargs before ``load_job()`` is called."""
    hook = _load_kwargs_hooks.get(provider_name)
    if hook is not None:
        hook(load_kwargs, params)
