# AWS Braket

Run benchmarks on quantum computers through Amazon Braket.

## Prerequisites

- An [AWS account](https://aws.amazon.com)
- AWS credentials with Braket permissions
- (Optional) An S3 bucket for job results

## Setup

### 1. Get AWS Credentials

1. Log in to the [AWS Console](https://console.aws.amazon.com)
2. Navigate to **IAM > Users > Your User > Security credentials**
3. Create an access key

### 2. Configure Environment

Add to your `.env` file:

```bash
AWS_ACCESS_KEY_ID="<your-access-key>"
AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Optionally, set your default region:

```bash
AWS_DEFAULT_REGION="us-east-1"
```

## Discovering Devices

AWS Braket provides access to multiple hardware providers (IonQ, Rigetti, IQM) and simulators. Device availability changes frequently.

To see currently available devices:

```python
from qbraid.runtime import load_provider

provider = load_provider("braket")
for device in provider.get_devices():
    print(f"{device.id}: {device.status}")
```

Or check the [Braket Console](https://console.aws.amazon.com/braket) for current devices and their ARNs.

## Usage

### Dispatch to Braket Hardware

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider braket \
    --device "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
```

### Dispatch to Braket Simulator

```bash
mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
    --provider braket \
    --device "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
```

### Poll Results

```bash
mgym job poll <JOB_ID>
```

## Pricing

See [AWS Braket Pricing](https://aws.amazon.com/braket/pricing/) for current rates.

## Troubleshooting

### Authentication Errors

Verify credentials:

```python
import boto3
client = boto3.client('braket')
print(client.search_devices(filters=[]))
```

### Region Issues

Some devices are only available in specific regions. Ensure your `AWS_DEFAULT_REGION` matches the device location.

### Insufficient Permissions

Your IAM user/role needs the `AmazonBraketFullAccess` policy or equivalent permissions.
