import importlib


def test_quantinuum_provider_import_only():
    # Ensure module and class can be imported without optional deps installed
    mod = importlib.import_module("metriq_gym.quantinuum.provider")
    cls = getattr(mod, "QuantinuumProvider")
    provider = cls()
    assert hasattr(provider, "get_devices")
    assert hasattr(provider, "get_device")

