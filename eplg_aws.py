"""EPLG benchmark for AWS Braket (IQM devices and simulators)."""
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
import numpy as np
import rustworkx as rx
import random
import json
import warnings
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def eplg_score_at_lengths(chain_lens, chain_eplgs, targets=[10, 20, 50, 100]):
    idx = {L: i for i, L in enumerate(chain_lens)}
    picked_vals, picks = [], []
    for t in targets:
        if t in idx:
            picked_vals.append(chain_eplgs[idx[t]])
            picks.append((t, t))
        else:
            # Nearest-neighbor fallback (in case a target isn't present)
            nearest = min(chain_lens, key=lambda L: (abs(L - t), L))
            picked_vals.append(chain_eplgs[idx[nearest]])
            picks.append((t, nearest))
    score = sum(picked_vals) / len(picked_vals)
    return score, picked_vals, picks


def to_edges(path):
    """Convert path of nodes to list of edges."""
    edges = []
    prev_node = None
    for node in path:
        if prev_node is not None:
            edges.append((prev_node, node))
        prev_node = node
    return edges


def create_complete_graph(num_qubits):
    """Create a complete graph (all-to-all connectivity) for IQM devices."""
    G = rx.PyDiGraph()
    # Add nodes
    for i in range(num_qubits):
        G.add_node(i)
    # Add all possible edges (complete graph)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            G.add_edge(i, j, None)
            G.add_edge(j, i, None)  # bidirectional
    return G


def random_chain_complete_graph(num_qubits, length, seed=None):
    """
    Sample a random chain from a complete graph.
    Since it's all-to-all connected, we can just pick random qubits.
    """
    rng = random.Random(seed)
    if length > num_qubits:
        raise ValueError(f"Chain length {length} cannot exceed number of qubits {num_qubits}")

    # For complete graph, any ordering works - just pick random qubits
    qubit_chain = rng.sample(range(num_qubits), length)
    return qubit_chain


# Suppress warnings and errors from qiskit_experiments
warnings.filterwarnings('ignore')
logging.getLogger('qiskit_experiments').setLevel(logging.CRITICAL)
os.environ['QISKIT_EXPERIMENTS_SKIP_SERVICE'] = '1'

# ============================================================================
# AWS Braket Device Configuration
# ============================================================================

from braket.devices import LocalSimulator
from braket.aws import AwsDevice

# Device configuration - Start with local simulator for testing
# Change to IQM device ARN for real hardware:
# - "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"  (20 qubits)
# - "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald" (5 qubits)

USE_AWS_DEVICE = False  # Set to True to use AWS devices (requires AWS credentials and S3 bucket)
USE_NOISE = True  # Set to True for noisy simulation

if USE_AWS_DEVICE:
    if USE_NOISE:
        # Use DM1 (density matrix simulator) which supports noise
        DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
        DEVICE_NAME = "DM1 (noisy)"
        NUM_QUBITS = 10  # DM1 supports up to 17 qubits
    else:
        # Use IQM Garnet device
        DEVICE_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
        DEVICE_NAME = "IQM Garnet"
        NUM_QUBITS = 20

    # Set AWS region for IQM devices
    import boto3
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
    boto_session = boto3.Session(region_name=AWS_REGION)
    from braket.aws import AwsSession
    aws_session = AwsSession(boto_session=boto_session)
    device = AwsDevice(DEVICE_ARN, aws_session=aws_session)
else:
    # Use local simulator
    if USE_NOISE:
        # Use LocalSimulator with density matrix backend for noise
        from braket.devices import LocalSimulator
        device = LocalSimulator("braket_dm")
        DEVICE_NAME = "LocalSimulator (DM, noisy)"
        NUM_QUBITS = 10
    else:
        # Use LocalSimulator (noiseless, free, no AWS credentials needed)
        device = LocalSimulator()
        DEVICE_NAME = "LocalSimulator"
        NUM_QUBITS = 10

# LayerFidelity requires a Clifford gate - use CZ
TWO_QUBIT_GATE = "cz"
ONE_QUBIT_BASIS_GATES = ["rz", "rx", "x"]

print(f"Device: {DEVICE_NAME}")
print(f"  Qubits: {NUM_QUBITS}")
print(f"  Two-qubit gate: {TWO_QUBIT_GATE}")
print(f"  One-qubit gates: {ONE_QUBIT_BASIS_GATES}")
print(f"  Topology: Complete graph (all-to-all connectivity)")
print(f"  Noise: {'Enabled' if USE_NOISE else 'Disabled'}")

# ============================================================================
# Chain Configuration
# ============================================================================

num_qubits_in_chain = 10

# Create complete graph for all-to-all connectivity
G = create_complete_graph(NUM_QUBITS)
print(f"\nCreated complete graph with {G.num_nodes()} nodes and {G.num_edges()} edges")

# Generate random chain
seed = 12345
qubit_chain = random_chain_complete_graph(NUM_QUBITS, num_qubits_in_chain, seed=seed)
print(f"Qubit chain: {qubit_chain}")

# Decompose the chain into two disjoint layers
all_pairs = to_edges(qubit_chain)
two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]
print(f"Layer 1: {two_disjoint_layers[0]}")
print(f"Layer 2: {two_disjoint_layers[1]}")

# ============================================================================
# LayerFidelity Experiment Setup
# ============================================================================

lfexp = LayerFidelity(
    physical_qubits=qubit_chain,
    two_qubit_layers=two_disjoint_layers,
    lengths=[2, 4],                # Start with shorter lengths
    num_samples=3,                 # Fewer samples for efficiency
    seed=42,
    two_qubit_gate=TWO_QUBIT_GATE,
    one_qubit_basis_gates=ONE_QUBIT_BASIS_GATES,
)

# Set maximum number of circuits per job
lfexp.experiment_options.max_circuits = 144

# Generate all circuits to run
circuits = lfexp.circuits()
print(f"\nGenerated {len(circuits)} circuits")

nshots = 100

# ============================================================================
# Circuit Conversion and Execution
# ============================================================================

print(f"\nConverting {len(circuits)} circuits to Braket format...")
from qbraid import transpile as qb_transpile
from pytket.passes import DecomposeBoxes

# Convert and prepare circuits
braket_circuits = []
for i, qiskit_circuit in enumerate(circuits):
    # Use qbraid to transpile from qiskit to braket
    braket_circuit = qb_transpile(qiskit_circuit, "braket")
    braket_circuits.append(braket_circuit)

    if (i + 1) % 5 == 0 or i == len(circuits) - 1:
        print(f"  Converted {i+1}/{len(circuits)} circuits")

print(f"✓ All circuits converted to Braket format")

# Add noise model if using noise
if USE_NOISE:
    from braket.circuits import Circuit

    print("\nAdding noise model to circuits...")
    # Add depolarizing noise to simulate realistic hardware
    # This will make EPLG values non-zero
    noise_prob = 0.01  # 1% depolarizing noise per gate

    # Workaround for Braket SDK bug: manually insert noise using fluent API
    # Cannot use apply_gate_noise() due to SDK bug with Noise.target attribute
    noisy_circuits = []
    for original_circuit in braket_circuits:
        # Separate gates from measurements
        gate_instructions = []
        measurement_instructions = []

        for instruction in original_circuit.instructions:
            if instruction.operator.name == 'Measure':
                measurement_instructions.append(instruction)
            else:
                gate_instructions.append(instruction)

        # Create new circuit and add gates with noise
        noisy_circuit = Circuit()

        for instruction in gate_instructions:
            # Add the gate instruction
            noisy_circuit.add_instruction(instruction)

            # Add depolarizing noise on all target qubits
            target_qubits = list(instruction.target)
            for qubit in target_qubits:
                # Use fluent API to add depolarizing noise
                noisy_circuit.depolarizing(qubit, probability=noise_prob)

        # Add measurements at the end
        for instruction in measurement_instructions:
            noisy_circuit.add_instruction(instruction)

        noisy_circuits.append(noisy_circuit)

    braket_circuits = noisy_circuits
    print(f"✓ Added {noise_prob*100}% depolarizing noise after all gates")

# ============================================================================
# Execute circuits
# ============================================================================

print(f"\nExecuting {len(braket_circuits)} circuits with {nshots} shots each...")
print("Waiting for results...")

results = []
if isinstance(device, LocalSimulator):
    # Local simulator - execute directly
    for i, circuit in enumerate(braket_circuits):
        result = device.run(circuit, shots=nshots).result()
        results.append(result)
        if (i + 1) % 5 == 0 or i == len(braket_circuits) - 1:
            print(f"  Executed {i+1}/{len(braket_circuits)} circuits")
else:
    # AWS device - submit tasks and wait
    import time

    # Get S3 bucket info from environment
    s3_bucket = os.getenv("AWS_BRAKET_S3_BUCKET")
    s3_prefix = os.getenv("AWS_BRAKET_S3_PREFIX", "eplg-results")
    s3_folder = (s3_bucket, s3_prefix)

    if not s3_bucket:
        print("ERROR: AWS_BRAKET_S3_BUCKET environment variable not set")
        import sys
        sys.exit(1)

    print(f"Using S3 bucket: s3://{s3_bucket}/{s3_prefix}")

    # Submit all tasks
    tasks = []
    for i, circuit in enumerate(braket_circuits):
        task = device.run(circuit, s3_folder, shots=nshots)
        tasks.append(task)
        if (i + 1) % 5 == 0 or i == len(braket_circuits) - 1:
            print(f"  Submitted {i+1}/{len(braket_circuits)} tasks")

    # Wait for all tasks to complete
    print("\nWaiting for all tasks to complete...")
    for i, task in enumerate(tasks):
        status = task.state()
        while status not in ["COMPLETED", "FAILED", "CANCELLED"]:
            time.sleep(2)
            status = task.state()

        if status == "COMPLETED":
            results.append(task.result())
            if (i + 1) % 5 == 0 or i == len(tasks) - 1:
                print(f"  Completed {i+1}/{len(tasks)} tasks")
        else:
            print(f"  Task {i+1} failed with status: {status}")
            import sys
            sys.exit(1)

print("\n" + "="*60)
print("✓ Job completed!")
print("="*60)
print(f"Received {len(results)} circuit results")

# ============================================================================
# Process results and run EPLG analysis
# ============================================================================

from qiskit.result import Result as QiskitResult
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_experiments.framework import ExperimentData

print("\nProcessing results...")

# Convert Braket results to Qiskit Result format for analysis
experiment_results = []
all_counts_list = []

for i, result in enumerate(results):
    # Get measurement counts from Braket result
    measurements = result.measurements

    # Convert measurements to counts dictionary
    # Braket returns array of measurement outcomes
    counts_dict = {}
    for measurement in measurements:
        # Convert measurement array to bitstring
        bitstring = "".join(str(int(b)) for b in measurement)
        counts_dict[bitstring] = counts_dict.get(bitstring, 0) + 1

    all_counts_list.append(counts_dict)

    # Get the circuit metadata (includes composite_index needed for analysis)
    circuit_metadata = circuits[i].metadata

    # Create ExperimentResult object with metadata
    exp_result = ExperimentResult(
        shots=nshots,
        success=True,
        data=ExperimentResultData(counts=counts_dict),
        header={"name": f"circuit_{i}", "metadata": circuit_metadata}
    )
    experiment_results.append(exp_result)

# Create Qiskit Result object
qiskit_result = QiskitResult(
    backend_name=DEVICE_NAME,
    backend_version="1.0",
    job_id="eplg_job",
    success=True,
    results=experiment_results
)

# Create ExperimentData and add the result
exp_data = ExperimentData(experiment=lfexp)
exp_data.add_data(qiskit_result)

# Run analysis
print("Running EPLG analysis...")
exp_data.auto_save = False
exp_data.block_for_results()

# Check the data before analysis
print(f"Number of data items: {len(exp_data.data())}")

# Run analysis
analysis_result = lfexp.analysis.run(exp_data)
print(f"Analysis returned: {type(analysis_result)}")

# Wait for analysis to complete
import time
max_wait = 60
waited = 0
status = str(exp_data.analysis_status())
print(f"Initial analysis status: {status}")

while "RUNNING" in status and waited < max_wait:
    time.sleep(0.5)
    waited += 0.5
    status = str(exp_data.analysis_status())
    if int(waited) != int(waited - 0.5):  # Print every second
        print(f"Waiting for analysis... {waited:.0f}s - Status: {status}")

print(f"Final analysis status: {status}")

# Check for errors
if exp_data.analysis_errors():
    print(f"Analysis errors: {exp_data.analysis_errors()}")

# Get analysis results
df = exp_data.analysis_results(dataframe=True)
print("\nAnalysis results:")
print(df)
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns)}")

pfdf = df[(df.name == "ProcessFidelity")]
pfdf = pfdf.fillna({"value": 0})

# Compute LF by chain length
lf_sets, lf_qubits = two_disjoint_layers, qubit_chain
full_layer = [None] * (len(lf_sets[0]) + len(lf_sets[1]))
full_layer[::2] = lf_sets[0]
full_layer[1::2] = lf_sets[1]
full_layer = [(lf_qubits[0],)] + full_layer + [(lf_qubits[-1],)]

pfs = [pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], "value"] for qubits in full_layer]
pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
pfs[0] = pfs[0] ** 2
pfs[-1] = pfs[-1] ** 2

chain_lens = list(range(4, len(pfs), 2))

if len(chain_lens) == 0:
    print("\n" + "="*60)
    print("⚠️  Chain too short for EPLG analysis")
    print("="*60)
    print(f"Chain has only {len(pfs)} process fidelities")
    print(f"Need at least 5 for meaningful analysis")
    import sys
    sys.exit(0)

chain_fids = []
for length in chain_lens:
    w = length + 1
    fid_w = max(
        np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1])
        for s in range(len(pfs) - w + 1)
    )
    chain_fids.append(fid_w)

num_2q_gates = [length - 1 for length in chain_lens]
chain_eplgs = [
    1 - (fid ** (1 / num_2q)) for num_2q, fid in zip(num_2q_gates, chain_fids)
]
chain_eplgs_aws = [float(x) for x in list(chain_eplgs)]

devices = {DEVICE_NAME: (chain_lens, chain_eplgs_aws)}

rows = []
for name, (lens, eplgs) in devices.items():
    score, vals, picks = eplg_score_at_lengths(lens, eplgs)
    rows.append({
        "device": name,
        "EPLG-10":  vals[0],
        "EPLG-20":  vals[1],
        "EPLG-50":  vals[2],
        "EPLG-100": vals[3],
        "score_avg": score,
        "picks": picks,
    })

rows.sort(key=lambda r: r["score_avg"])

print("\n" + "="*60)
print("EPLG Benchmark Results")
print("="*60)
hdr = f"{'device':<15} {'EPLG-10':>10} {'EPLG-20':>10} {'EPLG-50':>10} {'EPLG-100':>10} {'score_avg':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['device']:<15} "
          f"{r['EPLG-10']*1e3:>10.6f} {r['EPLG-20']*1e3:>10.6f} {r['EPLG-50']*1e3:>10.6f} {r['EPLG-100']*1e3:>10.6f} "
          f"{r['score_avg']*1e3:>12.6f}")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
