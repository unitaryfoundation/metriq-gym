# IBM Quantum

Run benchmarks on IBM Quantum hardware through the IBM Quantum Platform.

## Prerequisites

- An [IBM Quantum](https://quantum.ibm.com) account
- An IBM Cloud API key

## Setup

### 1. Get Your API Key

1. Log in to [IBM Cloud](https://cloud.ibm.com)
2. Navigate to **Manage > Access (IAM) > API keys**
3. Create a new API key or use an existing one

### 2. Configure Environment

Add to your `.env` file:

```bash
# Required
QISKIT_IBM_TOKEN="<your-ibm-cloud-api-key>"

# Optional
QISKIT_IBM_CHANNEL="ibm_quantum_platform"  # or "ibm_quantum"
QISKIT_IBM_INSTANCE="<instance-crn>"       # for specific instance access
```

## Available Devices

IBM Quantum devices vary by access level. Common devices include:

| Device | Qubits | Type |
|--------|--------|------|
| `ibm_fez` | 156 | Heron r2 |
| `ibm_marrakesh` | 156 | Heron r2 |
| `ibm_torino` | 133 | Heron r1 |
| `ibm_sherbrooke` | 127 | Eagle r3 |
| `ibm_brisbane` | 127 | Eagle r3 |

!!! note
    Device availability depends on your IBM Quantum plan. Check the [IBM Quantum Platform](https://quantum.ibm.com) for current device status.

## Usage

### Dispatch to IBM Hardware

```bash
mgym job dispatch metriq_gym/schemas/examples/bseq.example.json \
    --provider ibm --device ibm_sherbrooke
```

### Example Output

```
Starting dispatch on ibm:ibm_sherbrooke...
Dispatching BSEQ benchmark from bseq.example.json on ibm_sherbrooke...
Job dispatched with ID: 93a06a18-41d8-475a-a030-339fbf3accb9
```

### Poll Results

```bash
mgym job poll 93a06a18-41d8-475a-a030-339fbf3accb9
```

If the job is queued:
```
Job is queued at position 5. Please try again later.
```

When complete:
```
BSEQResult(largest_connected_size=100, fraction_connected=0.7874)
```

## Noise Model Simulation

Run benchmarks locally with IBM device noise models:

```bash
mgym job dispatch config.json --provider local --device ibm_sherbrooke
```

This uses Qiskit Aer with the noise model from the specified IBM device.

## Resource Estimation

Estimate job resources before dispatch:

```bash
mgym job estimate metriq_gym/schemas/examples/bseq.example.json \
    --provider ibm --device ibm_fez
```

## Troubleshooting

### Authentication Errors

Ensure your API key is valid and has appropriate permissions:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum_platform")
print(service.backends())
```

### Device Unavailable

Check device status on the IBM Quantum Platform. Devices may be:
- Under maintenance
- Reserved for specific access levels
- Temporarily offline

### Queue Times

IBM Quantum jobs may experience long queue times during peak usage. Consider:
- Using simulators for development
- Scheduling jobs during off-peak hours
- Using fair-share queuing through instance selection
