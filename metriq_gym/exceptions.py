class QBraidSetupError(Exception):
    pass


class DeviceCapacityError(ValueError):
    """Raised when a benchmark requests more qubits than a device provides."""
