import importlib


def test_origin_provider_import_only():
    mod = importlib.import_module("metriq_gym.origin.provider")
    provider_cls = getattr(mod, "OriginProvider")
    provider = provider_cls(api_key="dummy")
    assert hasattr(provider, "get_devices")
    assert hasattr(provider, "get_device")
