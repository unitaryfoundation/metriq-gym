# OriginQ (Wukong)

Run benchmarks on OriginQ Wukong superconducting quantum computers.

## Prerequisites

- An [OriginQ QCloud](https://account.originqc.com.cn/) account
- OriginQ API token
- (macOS only) `libidn2` library

## Setup

### 1. Get Your API Token

1. Log in to the [OriginQ Workbench](https://account.originqc.com.cn/)
2. Navigate to your account settings
3. Copy your API token

### 2. Configure Environment

Add to your `.env` file:

```bash
ORIGIN_API_KEY="<your-origin-api-token>"
```

### 3. macOS Setup

macOS users must install `libidn2` before using the OriginQ provider:

```bash
brew reinstall libidn2
```

Install this library **before** running `uv sync` or installing Metriq-Gym to avoid missing symbol errors during the `pyqpanda3` build.

## Available Devices

### Hardware

| Device ID | Alias | Description |
|-----------|-------|-------------|
| `WK_C102_400` | `origin_wukong` | 102-qubit Wukong hardware |
| `72` | - | 72-qubit hardware backend |

### Simulators

| Device ID | Max Qubits | Description |
|-----------|------------|-------------|
| `full_amplitude` | 35 | Full state vector simulation |
| `partial_amplitude` | 64 | Partial amplitude simulation |
| `single_amplitude` | 100 | Single amplitude simulation |

## Usage

### Dispatch to Wukong Hardware

```bash
# Using alias
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider origin --device origin_wukong

# Using numeric ID
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider origin --device 72
```

### Dispatch to OriginQ Simulator

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider origin --device full_amplitude
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

## Verify Available Devices

List devices accessible with your account:

```bash
uv run python -c "from metriq_gym.run import load_provider; provider = load_provider('origin'); print([device.id for device in provider.get_devices()])"
```

Or in Python:

```python
from metriq_gym.run import load_provider

provider = load_provider('origin')
devices = provider.get_devices()
for device in devices:
    print(device.id)
```

## Troubleshooting

### "resource is null" Error

This error indicates:
- The device name is incorrect
- Your account lacks permission for the requested device

Verify available devices with the listing command above.

### Missing libidn2 (macOS)

If you see symbol errors related to `pyqpanda3`:

```bash
# Install the library
brew reinstall libidn2

# Reinstall metriq-gym
pip install --force-reinstall metriq-gym
```

### Authentication Errors

Verify your API token is correctly set:

```python
import os
print(os.environ.get("ORIGIN_API_KEY"))
```

### Connection Issues

OriginQ services are based in China. Connection latency or timeouts may occur from other regions. Consider:
- Retrying failed requests
- Using longer timeout values
- Running during off-peak hours

## Device Topology

Wukong devices use a 2D grid topology. For topology-dependent benchmarks (BSEQ, Mirror Circuits), the provider automatically retrieves connectivity information.

```python
from metriq_gym.run import load_provider

provider = load_provider('origin')
device = provider.get_device('origin_wukong')
# Topology available via device.metadata
```
