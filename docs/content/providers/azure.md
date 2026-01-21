# Azure Quantum

Run benchmarks on quantum computers through Microsoft Azure Quantum.

## Prerequisites

- An [Azure account](https://azure.microsoft.com)
- An Azure Quantum workspace
- Azure Quantum connection string

## Setup

### 1. Create an Azure Quantum Workspace

1. Log in to the [Azure Portal](https://portal.azure.com)
2. Create a new **Azure Quantum** workspace
3. Add providers (IonQ, Quantinuum, Rigetti, etc.) to your workspace

### 2. Get Connection String

1. Navigate to your Azure Quantum workspace
2. Go to **Overview > Connection string**
3. Copy the connection string

### 3. Configure Environment

Add to your `.env` file:

```bash
AZURE_QUANTUM_CONNECTION_STRING="<your-connection-string>"
```

The connection string format is:
```
SubscriptionId=xxx;ResourceGroupName=xxx;WorkspaceName=xxx;Location=xxx
```

## Discovering Devices

Azure Quantum provides access to multiple hardware providers (IonQ, Quantinuum, Rigetti). Available devices depend on your workspace configuration.

To see currently available devices:

```python
from qbraid.runtime import load_provider

provider = load_provider("azure")
for device in provider.get_devices():
    print(f"{device.id}: {device.status}")
```

Or check your workspace in the [Azure Portal](https://portal.azure.com) under **Providers**.

!!! note
    Enable providers in your Azure Quantum workspace to access their devices.

## Usage

### Dispatch to Azure Quantum Hardware

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider azure --device ionq.qpu.aria-1
```

### Dispatch to Azure Quantum Simulator

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider azure --device ionq.simulator
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

## Pricing

Azure Quantum uses provider-specific pricing:
- IonQ: Per-shot pricing
- Quantinuum: HQC-based pricing
- Rigetti: Per-shot pricing

Check [Azure Quantum Pricing](https://azure.microsoft.com/pricing/details/azure-quantum/) and individual provider pricing for details.

## Azure Credits

Azure Quantum offers:
- Free credits for new users
- Academic programs
- Research grants

Check eligibility in the Azure Portal.

## Troubleshooting

### Connection Errors

Verify your connection string:

```python
from azure.quantum import Workspace

workspace = Workspace(
    subscription_id="...",
    resource_group="...",
    name="...",
    location="..."
)
print(workspace.get_targets())
```

### Provider Not Available

Ensure the provider is:

1. Enabled in your workspace
2. Available in your region
3. Accessible with your subscription tier

### Job Failures

Check job status in the Azure Portal under your workspace's **Job management** section for detailed error messages.
