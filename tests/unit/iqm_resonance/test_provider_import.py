import importlib


def test_iqm_resonance_provider_import_only():
    # Ensure module and class can be imported without a configured token
    mod = importlib.import_module("metriq_gym.iqm_resonance.provider")
    cls = getattr(mod, "IQMResonanceProvider")
    provider = cls()
    assert hasattr(provider, "get_devices")
    assert hasattr(provider, "get_device")
