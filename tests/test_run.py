import logging
import pytest
from unittest.mock import MagicMock, patch

from qbraid import QbraidError
from metriq_gym.run import setup_device
from metriq_gym.exceptions import QBraidSetupError


class FakeDevice:
    def __init__(self, id):
        self.id = id


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def mock_device():
    return FakeDevice(id="test_device")


@pytest.fixture
def patch_load_provider(mock_provider, monkeypatch):
    monkeypatch.setattr("metriq_gym.run.load_provider", lambda _: mock_provider)


def test_setup_device_success(mock_provider, mock_device, patch_load_provider):
    mock_provider.get_device.return_value = mock_device

    provider_name = "test_provider"
    backend_name = "test_backend"

    device = setup_device(provider_name, backend_name)

    mock_provider.get_device.assert_called_once_with(backend_name)
    assert device == mock_device


@patch("metriq_gym.run.get_providers")
def test_setup_device_invalid_provider(get_providers_patch, caplog):
    get_providers_patch.return_value = ["supported_provider"]
    caplog.set_level(logging.INFO)

    provider_name = "unsupported_provider"
    backend_name = "whatever_backend"

    with pytest.raises(QBraidSetupError, match="Provider not found"):
        setup_device(provider_name, backend_name)

    # Verify the printed output
    assert f"No provider matching the name '{provider_name}' found." in caplog.text
    assert "Providers available: ['supported_provider']" in caplog.text


def test_setup_device_invalid_device(mock_provider, patch_load_provider, caplog):
    caplog.set_level(logging.INFO)
    mock_provider.get_device.side_effect = QbraidError()
    mock_provider.get_devices.return_value = [FakeDevice(id="device1"), FakeDevice(id="device2")]

    provider_name = "test_provider"
    backend_name = "non_existent_backend"

    with pytest.raises(QBraidSetupError, match="Device not found"):
        setup_device(provider_name, backend_name)

    # Verify the printed output
    assert (
        f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        in caplog.text
    )
    assert "Devices available: ['device1', 'device2']" in caplog.text

class TestLocalProviderIntegration:
    """Test suite for LocalProvider integration - simulator agnostic."""

    @pytest.fixture
    def mock_local_device(self):
        """Generic mock device that represents any local simulator."""
        mock_device = MagicMock()
        mock_device.id = "generic.simulator.method"
        mock_device.device_type = "SIMULATOR"
        mock_device.num_qubits = 32
        mock_device.metadata.return_value = {
            'device_id': 'generic.simulator.method',
            'device_type': 'SIMULATOR',
            'num_qubits': 32,
            'status': 'ONLINE',
            'local': True,
            'simulator': True
        }
        return mock_device

    def test_setup_device_local_provider_universal_integration(self, mock_local_device, caplog):
        """Test E2E integration with local provider - works with ANY available simulator."""
        caplog.set_level(logging.INFO)
        
        with patch("metriq_gym.run.LocalProvider") as mock_local_provider_class:
            mock_local_provider = MagicMock()
            mock_local_provider.get_device.return_value = mock_local_device
            mock_local_provider_class.return_value = mock_local_provider
            
            # Test with any device name - should work universally
            device = setup_device("local", "any_simulator_name")
            
            # Verify universal integration flow
            mock_local_provider_class.assert_called_once()
            mock_local_provider.get_device.assert_called_once_with("any_simulator_name")
            assert device == mock_local_device
            
            # Verify success logging (universal for all simulators)
            assert "Successfully configured local device" in caplog.text

    def test_local_provider_universal_device_mapping_and_discovery(self, mock_local_device):
        """Test that LocalProvider works universally with any available simulators and mappings."""
        from metriq_gym.run import LocalProvider, DEVICE_MAPPING
        
        # Mock any available simulators (could be Qiskit, Qrack, QuEST, etc.)
        mock_available_simulators = {
            "simulator1.backend.method": "First available simulator",
            "simulator2.backend.method": "Second available simulator", 
            "qiskit.aer.automatic": "Qiskit if available",
        }
        
        with patch("metriq_gym.qplatform.device.get_available_simulators") as mock_get_sims, \
             patch("metriq_gym.qplatform.device.create_local_device") as mock_create_device:
            
            mock_get_sims.return_value = mock_available_simulators
            mock_create_device.return_value = mock_local_device
            
            provider = LocalProvider()
            
            # Test 1: Direct device spec (no mapping) - should work with any simulator
            if mock_available_simulators:
                first_available = next(iter(mock_available_simulators.keys()))
                device = provider.get_device(first_available)
                mock_create_device.assert_called_with(first_available)
                assert device == mock_local_device
            
            # Test 2: Legacy mapping (backward compatibility) - universal behavior
            if DEVICE_MAPPING:
                first_legacy = next(iter(DEVICE_MAPPING.keys()))
                expected_mapped = DEVICE_MAPPING[first_legacy]
                
                mock_create_device.reset_mock()
                device = provider.get_device(first_legacy)
                mock_create_device.assert_called_with(expected_mapped)
                assert device == mock_local_device
            
            # Test 3: get_devices() - should work with any number of available simulators
            devices = provider.get_devices()
            assert len(devices) <= len(mock_available_simulators)  # Some might fail gracefully
            assert all(d == mock_local_device for d in devices)

    def test_local_provider_error_handling_and_remote_preservation(self, mock_provider, mock_device, patch_load_provider, caplog):
        """Test error handling for local provider and ensure remote functionality preserved."""
        caplog.set_level(logging.ERROR)
        
        # Test 1: Local provider error handling (universal for any simulator)
        mock_empty_simulators = {}
        with patch("metriq_gym.qplatform.device.get_available_simulators") as mock_get_sims:
            mock_get_sims.return_value = mock_empty_simulators
            
            with pytest.raises(QBraidSetupError, match="Local device .* not found"):
                setup_device("local", "nonexistent_simulator")
            
            # Should log error with available options (adaptive to environment)
            assert "Local device 'nonexistent_simulator' not supported" in caplog.text
        
        # Test 2: Regression test - remote providers must still work (universal)
        mock_provider.get_device.return_value = mock_device
        
        # Test with any remote provider name (not "local")
        remote_device = setup_device("any_remote_provider", "any_remote_device")
        
        # Should use existing remote provider logic unchanged
        mock_provider.get_device.assert_called_once_with("any_remote_device")
        assert remote_device == mock_device
        
        # Test 3: Exception handling for local provider (universal)
        with patch("metriq_gym.run.LocalProvider") as mock_local_provider_class:
            mock_local_provider_class.side_effect = Exception("Generic simulator error")
            
            with pytest.raises(QBraidSetupError, match="Local device setup failed"):
                setup_device("local", "any_simulator")
            
            # Should log appropriate error (simulator agnostic)
            assert "Failed to setup local device" in caplog.text