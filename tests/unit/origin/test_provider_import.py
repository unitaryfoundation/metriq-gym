import importlib

import pytest


pytest.importorskip("pyqpanda3", reason="Origin provider requires pyqpanda3")


def test_origin_provider_import_only():
    mod = importlib.import_module("metriq_gym.origin.provider")
    provider_cls = getattr(mod, "OriginProvider")
    provider = provider_cls(api_key="dummy")
    assert hasattr(provider, "get_devices")
    assert hasattr(provider, "get_device")
