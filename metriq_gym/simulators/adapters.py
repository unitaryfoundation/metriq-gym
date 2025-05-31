
"""Simulator adapters for local quantum simulators.

This module provides a plugin architecture for supporting multiple local quantum simulators
through standardized adapters. Each adapter implements the SimulatorAdapter protocol to
provide a unified interface for different simulator backends.
"""

import json
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol

from qbraid.runtime.result_data import GateModelResultData, MeasCount


class SimulatorAdapter(Protocol):
    """Protocol defining the interface for simulator adapters.
    
    All simulator adapters must implement these methods to provide
    a standardized interface for running circuits and getting backend information.
    """
    
    def run_circuits(self, circuits, shots: int) -> Any:
        """Run circuits on the simulator backend.
        
        Args:
            circuits: Quantum circuit(s) to execute
            shots: Number of measurement shots
            
        Returns:
            Raw results from the simulator
        """
        ...
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information (num_qubits, basis_gates, etc.).
        
        Returns:
            Dictionary containing backend configuration information
        """
        ...
    
    def get_version(self) -> str:
        """Get simulator version.
        
        Returns:
            Version string for the simulator
        """
        ...


class QiskitAerAdapter:
    """Adapter for Qiskit Aer simulators.
    
    This adapter provides a standardized interface for Qiskit Aer simulators,
    supporting multiple simulation methods and configurations.
    """
    
    def __init__(self, method: str = "automatic"):
        """Initialize Qiskit Aer adapter.
        
        Args:
            method: Simulation method ('automatic', 'statevector', 'stabilizer', etc.)
            
        Raises:
            ImportError: If Qiskit Aer is not available
        """
        try:
            from qiskit_aer import AerSimulator
            
            if method == "automatic":
                self._backend = AerSimulator()
            else:
                self._backend = AerSimulator(method=method)
                
            self.method = method
            
        except ImportError:
            raise ImportError(
                "Qiskit Aer is required for Qiskit simulators. "
                "Install with: pip install qiskit-aer"
            )
    
    def run_circuits(self, circuits, shots: int):
        """Run circuits on Qiskit Aer.
        
        Args:
            circuits: Quantum circuit(s) to execute
            shots: Number of measurement shots
            
        Returns:
            List of Qiskit result objects
        """
        if isinstance(circuits, list):
            results = []
            for circuit in circuits:
                result = self._backend.run(circuit, shots=shots).result()
                results.append(result)
            return results
        else:
            return [self._backend.run(circuits, shots=shots).result()]
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get Qiskit backend information.
        
        Returns:
            Dictionary containing backend configuration
        """
        config = self._backend.configuration()
        return {
            "num_qubits": getattr(config, "n_qubits", 64),
            "basis_gates": getattr(config, "basis_gates", [
                "u1", "u2", "u3", "cx", "id", "x", "y", "z", "h", "s", "t", "sdg", "tdg"
            ]),
            "coupling_map": getattr(config, "coupling_map", None),
            "method": self.method,
            "backend_name": config.backend_name,
        }
    
    def get_version(self) -> str:
        """Get Qiskit Aer version.
        
        Returns:
            Qiskit Aer version string
        """
        try:
            import qiskit_aer
            return qiskit_aer.__version__
        except (ImportError, AttributeError):
            return "unknown"


class LocalDevice:
    """Local quantum device wrapper using adapter pattern.
    
    This class provides a qBraid-compatible interface for local quantum simulators
    through pluggable adapters, making it extensible to multiple simulator backends.
    
    Attributes:
        id (str): Device identifier
        device_type (str): Type of device ("SIMULATOR")
        num_qubits (int): Number of qubits available
        profile: Device profile with configuration information
    """
    
    def __init__(self, device_id: str, adapter: SimulatorAdapter):
        """Initialize a local device with a simulator adapter.
        
        Args:
            device_id: The identifier for the local device
            adapter: The simulator adapter to use
        """
        self.id = device_id
        self.device_type = "SIMULATOR"
        self._adapter = adapter
        self._setup_device()
        
        # Store results cache directory for this session
        self._results_cache = tempfile.mkdtemp(prefix="metriq_local_")
    
    def _setup_device(self):
        """Setup device properties from adapter."""
        backend_info = self._adapter.get_backend_info()
        self.num_qubits = backend_info.get("num_qubits", 64)
        self.profile = LocalDeviceProfile(backend_info)
    
    def run(self, circuits, shots: int = 1000):
        """Run quantum circuits on the local simulator.
        
        Args:
            circuits: Quantum circuit(s) to execute
            shots: Number of measurement shots
            
        Returns:
            LocalJob: A job object containing the execution results
            
        Raises:
            RuntimeError: If simulator execution fails
        """
        from metriq_gym.qplatform.job import LocalJob
        
        # Execute circuits immediately (synchronous)
        try:
            results = self._adapter.run_circuits(circuits, shots)
            
            # Create a local job with immediate results
            job_id = str(uuid.uuid4())
            return LocalJob(job_id, results, self._results_cache, self._adapter)
            
        except Exception as e:
            raise RuntimeError(f"Local simulator execution failed: {e}")
    
    def metadata(self):
        """Get device metadata.
        
        Returns:
            dict: Device metadata including type, qubits, and status
        """
        backend_info = self._adapter.get_backend_info()
        return {
            "device_id": self.id,
            "device_type": self.device_type,
            "num_qubits": self.num_qubits,
            "status": "ONLINE",
            "queue_depth": 0,
            "local": True,
            "simulator": True,
            "method": backend_info.get("method", "unknown"),
            "backend_name": backend_info.get("backend_name", "local_simulator"),
        }


class LocalDeviceProfile:
    """Profile for local device containing configuration information.
    
    This class stores device profile information extracted from simulator
    backend configuration for use by device introspection functions.
    """
    
    def __init__(self, backend_info: Dict[str, Any]):
        """Initialize profile from backend info.
        
        Args:
            backend_info: Backend information dictionary from adapter
        """
        self.basis_gates = backend_info.get("basis_gates", [
            "u1", "u2", "u3", "cx", "id", "x", "y", "z", "h", "s", "t", "sdg", "tdg"
        ])
        self.coupling_map = backend_info.get("coupling_map", None)
        self.n_qubits = backend_info.get("num_qubits", 64)
        self.method = backend_info.get("method", "automatic")
        self.local = True
        self.simulator = True


# Registry of available simulator adapters
SIMULATOR_ADAPTERS = {
    "qiskit.aer": QiskitAerAdapter,
}


def get_available_simulators() -> Dict[str, str]:
    """Get available local simulators.
    
    This function dynamically discovers which simulators are available
    in the current environment by checking for required packages.
    
    Returns:
        Dict mapping device IDs to descriptions
    """
    available = {}
    
    # Check Qiskit Aer
    try:
        from qiskit_aer import AerSimulator
        available.update({
            "qiskit.aer.automatic": "Qiskit Aer automatic method selection",
            "qiskit.aer.statevector": "Qiskit Aer statevector simulator",
            "qiskit.aer.stabilizer": "Qiskit Aer stabilizer simulator", 
            "qiskit.aer.density_matrix": "Qiskit Aer density matrix simulator",
        })
    except ImportError:
        pass
    
    return available


def create_local_device(device_spec: str) -> LocalDevice:
    """Factory function to create local devices based on specification.
    
    Args:
        device_spec: Device specification like 'qiskit.aer.statevector'
        
    Returns:
        LocalDevice: Configured local device
        
    Raises:
        ValueError: If device specification is invalid or unsupported
    """
    parts = device_spec.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid device spec: {device_spec}. Expected format: 'backend.simulator[.method]'")
    
    backend = parts[0]
    simulator = parts[1]
    method = parts[2] if len(parts) > 2 else "automatic"
    
    adapter_key = f"{backend}.{simulator}"
    if adapter_key not in SIMULATOR_ADAPTERS:
        raise ValueError(f"Unsupported simulator: {adapter_key}")
    
    adapter_class = SIMULATOR_ADAPTERS[adapter_key]
    adapter = adapter_class(method)
    
    return LocalDevice(device_spec, adapter)

