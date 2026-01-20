# Provider Configuration

Metriq-Gym supports multiple quantum hardware providers through a unified interface built on [qBraid Runtime](https://docs.qbraid.com/runtime/).

## About qBraid Runtime

Metriq-Gym uses [qBraid Runtime](https://docs.qbraid.com/runtime/) as an abstraction layer for quantum hardware access. This provides:

- **Unified API**: Same code works across IBM, IonQ, AWS Braket, and Azure Quantum
- **Automatic transpilation**: Circuits are converted to provider-native formats
- **Credential management**: Consistent environment variable configuration

!!! note "Impact on Benchmarks"
    qBraid handles circuit transpilation, which may differ from provider-native SDKs. For most benchmarks, this has negligible impact. However, performance-sensitive measurements like CLOPS should note that results reflect the qBraid pipeline, not native SDK performance.

## Supported Providers

| Provider | Provider ID | Description |
|----------|-------------|-------------|
| [IBM Quantum](ibm.md) | `ibm` | IBM's quantum computers and simulators |
| [IonQ](ionq.md) | `ionq` | IonQ trapped-ion quantum computers |
| [AWS Braket](braket.md) | `braket` | Amazon Braket quantum computing service |
| [Azure Quantum](azure.md) | `azure` | Microsoft Azure Quantum |
| [Quantinuum](quantinuum.md) | `quantinuum` | Quantinuum H-series trapped-ion systems |
| [OriginQ](originq.md) | `origin` | OriginQ Wukong superconducting systems |
| [Local Simulators](local.md) | `local` | Qiskit Aer local simulation |

## Provider Architecture

Metriq-Gym uses two types of provider integrations:

### qBraid-Managed Providers

IBM, IonQ, Braket, and Azure use qBraid Runtime's native provider support:

```python
from qbraid.runtime import load_provider
provider = load_provider("ibm")
```

### Custom Providers

Local, Quantinuum (NEXUS), and OriginQ use custom provider implementations registered via entry points:

```python
# Registered in pyproject.toml
[project.entry-points."qbraid.providers"]
local = "metriq_gym.local.provider:LocalProvider"
quantinuum = "metriq_gym.quantinuum.provider:QuantinuumProvider"
origin = "metriq_gym.origin.provider:OriginProvider"
```

## Credential Configuration

Create a `.env` file in your project root with provider credentials:

```bash
# Copy the template
cp .env.example .env

# Edit with your credentials
nano .env
```

!!! warning "Protect Your Credentials"
    Never commit `.env` files to version control. Ensure `.env` is in your `.gitignore` file.

### Environment Variables Reference

| Variable | Provider | Description |
|----------|----------|-------------|
| `QISKIT_IBM_TOKEN` | IBM | IBM Cloud API key |
| `QISKIT_IBM_CHANNEL` | IBM | Channel type (optional) |
| `QISKIT_IBM_INSTANCE` | IBM | Instance CRN (optional) |
| `IONQ_API_KEY` | IonQ | IonQ API key |
| `AWS_ACCESS_KEY_ID` | Braket | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Braket | AWS secret key |
| `AZURE_QUANTUM_CONNECTION_STRING` | Azure | Azure connection string |
| `QUANTINUUM_NEXUS_USERNAME` | Quantinuum | NEXUS username |
| `QUANTINUUM_NEXUS_PASSWORD` | Quantinuum | NEXUS password |
| `ORIGIN_API_KEY` | OriginQ | OriginQ API token |

## Device Selection

List available devices for a provider:

```python
from metriq_gym.run import load_provider

provider = load_provider("ibm")
devices = provider.get_devices()
for device in devices:
    print(f"{device.id}: {device.status}")
```

Via CLI, use provider-specific documentation to find device names.

## Next Steps

- [IBM Quantum](ibm.md) - IBM Quantum Platform setup
- [IonQ](ionq.md) - IonQ Cloud setup
- [AWS Braket](braket.md) - Amazon Braket setup
- [Azure Quantum](azure.md) - Azure Quantum setup
- [Quantinuum](quantinuum.md) - Quantinuum NEXUS setup
- [OriginQ](originq.md) - OriginQ Wukong setup
- [Local Simulators](local.md) - Running benchmarks locally
