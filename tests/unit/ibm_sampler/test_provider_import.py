"""Verify that the ibm_sampler provider can be imported and discovered."""

import importlib


def test_ibm_sampler_provider_import():
    mod = importlib.import_module("metriq_gym.ibm_sampler.provider")
    cls = getattr(mod, "IBMSamplerProvider")
    assert hasattr(cls, "get_devices")
    assert hasattr(cls, "get_device")


def test_ibm_sampler_device_import():
    mod = importlib.import_module("metriq_gym.ibm_sampler.device")
    cls = getattr(mod, "IBMSamplerDevice")
    assert hasattr(cls, "submit")


def test_ibm_sampler_in_registered_providers():
    from qbraid.runtime import get_providers

    providers = get_providers()
    assert "ibm_sampler" in providers
