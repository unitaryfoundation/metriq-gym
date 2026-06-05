import os

import pytest

from metriq_gym.local import provider as provider_mod


def test_local_provider_resolves_ibm_style_fake_backend_alias(monkeypatch):
    for key in [
        "QISKIT_IBM_TOKEN",
        "QISKIT_IBM_CHANNEL",
        "QISKIT_IBM_INSTANCE",
        "IBM_QUANTUM_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
        assert os.environ.get(key) is None

    fake_backend_names = sorted(
        backend.name
        for backend in provider_mod.FakeProviderForBackendV2().backends()
        if backend.name.startswith("fake_")
    )
    if not fake_backend_names:
        pytest.skip("No local fake_* backends available from Qiskit fake_provider")

    fake_backend_name = (
        "fake_torino" if "fake_torino" in fake_backend_names else fake_backend_names[0]
    )
    ibm_style_device_id = fake_backend_name.replace("fake_", "ibm_", 1)

    class RuntimeServiceShouldNotBeNeeded:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "QiskitRuntimeService should not be required for local fake backend aliases"
            )

    monkeypatch.setattr(
        provider_mod, "QiskitRuntimeService", RuntimeServiceShouldNotBeNeeded
    )

    device = provider_mod.LocalProvider().get_device(ibm_style_device_id)

    assert device.id == ibm_style_device_id
    assert device.profile.simulator is True
    assert device.profile.extra["backend"] is not None
