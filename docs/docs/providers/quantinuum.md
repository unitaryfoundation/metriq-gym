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

## Available Devices

### Hardware

| Device | Qubits | Description |
|--------|--------|-------------|
| `H1-1` | 20 | H1 generation system |
| `H2-1` | 56 | H2 generation system |

### Emulators

| Device | Description |
|--------|-------------|
| `H1-1E` | H1-1 emulator with realistic noise |
| `H2-1E` | H2-1 emulator with realistic noise |

### Syntax Checkers

| Device | Description |
|--------|-------------|
| `H1-1SC` | H1-1 syntax validation (no execution) |
| `H2-1SC` | H2-1 syntax validation (no execution) |

!!! tip
    Use syntax checkers to validate circuits before running on hardware. They're free and instant.

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
