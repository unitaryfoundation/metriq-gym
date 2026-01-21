# Quantinuum

Run benchmarks on Quantinuum H-series trapped-ion quantum computers through the NEXUS platform.

## Prerequisites

- A [Quantinuum NEXUS](https://nexus.quantinuum.com) account
- NEXUS credentials (username and password)

## Setup

### 1. Get NEXUS Credentials

1. Register at [Quantinuum NEXUS](https://nexus.quantinuum.com)
2. Note your username and password

### 2. Configure Environment

Add to your `.env` file:

```bash
QUANTINUUM_NEXUS_USERNAME="<your-username>"
QUANTINUUM_NEXUS_PASSWORD="<your-password>"
QUANTINUUM_NEXUS_PROJECT_NAME="metriq-gym"

# Optional: optimization level for compilation (default: 1)
QUANTINUUM_NEXUS_OPT_LEVEL="1"
```

### 3. Initial Login

The first time you use Quantinuum, you must complete a manual login workflow:

```bash
qnx login
```

Follow the prompts to authenticate and link your account.

## Discovering Devices

Quantinuum offers H-series hardware, emulators, and syntax checkers. To see currently available devices:

```python
from qbraid.runtime import load_provider

provider = load_provider("quantinuum")
for device in provider.get_devices():
    print(f"{device.id}: {device.status}")
```

Or check the [NEXUS portal](https://nexus.quantinuum.com) for current device availability.

!!! tip
    Use syntax checkers (devices ending in `SC`) to validate circuits before running on hardware. They're free and instant.

## Usage

### Dispatch to Quantinuum Hardware

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider quantinuum --device H1-1
```

### Dispatch to Emulator

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider quantinuum --device H1-1E
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

## H-series Quantum Credits (HQCs)

Quantinuum uses HQCs for billing. The formula is:

```
HQC = 5 + C * (N1 + 10*N2 + 5*Nm) / 5000
```

Where:
- `C` = number of shots
- `N1` = single-qubit gate count
- `N2` = two-qubit gate count
- `Nm` = measurement count

### Estimate HQCs Before Running

```bash
mgym job estimate metriq_gym/schemas/examples/wit.example.json \
    --provider quantinuum
```

Example output:
```
Quantinuum HQC estimate: 52.4
```

## NEXUS Projects

Jobs are organized into NEXUS projects. Set the project name:

```bash
QUANTINUUM_NEXUS_PROJECT_NAME="my-project"
```

## Compilation Optimization

Control TKET compilation optimization:

```bash
# Levels: 0 (none), 1 (default), 2 (aggressive)
QUANTINUUM_NEXUS_OPT_LEVEL="2"
```

Higher levels may reduce gate counts but increase compilation time.

## Troubleshooting

### Login Required

If you see authentication errors, re-run:

```bash
qnx login
```

### HQC Limits

Check your HQC balance in the NEXUS portal. Jobs exceeding your balance will fail.

### Compilation Errors

Some circuits may not compile efficiently for trapped-ion hardware. Try:
- Reducing circuit depth
- Using optimization level 2
- Checking for unsupported operations
