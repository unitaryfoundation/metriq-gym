# IonQ

Run benchmarks on IonQ trapped-ion quantum computers.

## Prerequisites

- An [IonQ](https://ionq.com) account
- An IonQ API key

## Setup

### 1. Get Your API Key

1. Log in to [IonQ Cloud](https://cloud.ionq.com)
2. Navigate to **Settings > API Keys**
3. Create a new API key

### 2. Configure Environment

Add to your `.env` file:

```bash
IONQ_API_KEY="<your-ionq-api-key>"
```

## Discovering Devices

Device availability changes frequently. To see currently available devices:

```python
from qbraid.runtime import load_provider

provider = load_provider("ionq")
for device in provider.get_devices():
    print(f"{device.id}: {device.num_qubits} qubits - {device.status}")
```

Or check the [IonQ Cloud dashboard](https://cloud.ionq.com) for current device status.

## Usage

### Dispatch to IonQ Hardware

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider ionq --device ionq_aria
```

### Dispatch to IonQ Simulator

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider ionq --device ionq_simulator
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

## IonQ-Specific Considerations

### Native Gate Set

IonQ uses a native gate set of single-qubit rotations and the MS (Molmer-Sorensen) two-qubit gate. Circuits are automatically transpiled.

### Debiasing

IonQ supports debiasing options for error mitigation. Configure via provider settings if available.

### Pricing

See [IonQ Pricing](https://ionq.com/pricing) for current rates.

## Troubleshooting

### Authentication Errors

Verify your API key:

```python
import os
os.environ["IONQ_API_KEY"] = "your-key"

from qbraid.runtime import load_provider
provider = load_provider("ionq")
print(provider.get_devices())
```

### Device Unavailable

Check device status on the IonQ Cloud dashboard. Devices may be under maintenance.
