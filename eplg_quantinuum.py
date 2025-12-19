"""EPLG benchmark for Quantinuum devices via NEXUS platform."""
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
    """Create a complete graph (all-to-all connectivity) for trapped-ion devices."""
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


# Suppress warnings and errors from qiskit_experiments about deprecated experiment service
warnings.filterwarnings('ignore')
logging.getLogger('qiskit_experiments').setLevel(logging.CRITICAL)
os.environ['QISKIT_EXPERIMENTS_SKIP_SERVICE'] = '1'

# ============================================================================
# Quantinuum Device Configuration
# ============================================================================

# Import qnexus for Quantinuum access
import qnexus

# Device configuration
DEVICE_NAME = "H1-1LE"  # H1 Light Emulator (free, no cost)
NUM_QUBITS = 20  # H1 series has 20 qubits
# LayerFidelity requires a Clifford gate - use CZ which can be compiled to Quantinuum's native ZZ
TWO_QUBIT_GATE = "cz"  # CZ gate (Clifford, will compile to native gates)
ONE_QUBIT_BASIS_GATES = ["rz", "rx", "x"]  # Quantinuum native 1-qubit gates

print(f"Device: {DEVICE_NAME} (H1 Light Emulator - free)")
print(f"  Qubits: {NUM_QUBITS}")
print(f"  Two-qubit gate: {TWO_QUBIT_GATE}")
print(f"  One-qubit gates: {ONE_QUBIT_BASIS_GATES}")
print(f"  Topology: Complete graph (all-to-all connectivity)")

# ============================================================================
# Chain Configuration
# ============================================================================

# Regular parameters
num_qubits_in_chain = 10  # Start smaller for Quantinuum (costs per shot)

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
    num_samples=3,                 # Fewer samples for cost efficiency
    seed=42,
    # NOTE: Do NOT provide backend for Quantinuum
    # Instead provide the gate set explicitly:
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
# Execution
# ============================================================================

print("\nConnecting to NEXUS...")

# Get the metriq-gym project
projects = list(qnexus.projects.get_all())
if not projects:
    print("No projects found. Create one at https://nexus.quantinuum.com/")
    import sys
    sys.exit(1)

# Find metriq-gym project
PROJECT_NAME = "metriq-gym"
project = None
for p in projects:
    p_name = p.annotations.name if hasattr(p.annotations, 'name') else None
    if p_name == PROJECT_NAME:
        project = p
        break

if project is None:
    print(f"Project '{PROJECT_NAME}' not found")
    print("Available projects:")
    for p in projects:
        p_name = p.annotations.name if hasattr(p.annotations, 'name') else "Unknown"
        print(f"  - {p_name}")
    import sys
    sys.exit(1)

print(f"Using project: {PROJECT_NAME} ({project.id})")

# Get all available devices
all_devices = qnexus.devices.get_all()
device_list = list(all_devices)
print(f"Found {len(device_list)} devices")

# Find the H1 Light Emulator
device = None
for d in device_list:
    print(f"  - {d.device_name}")
    if d.device_name == DEVICE_NAME:
        device = d
        break

if device is None:
    print(f"\nDevice '{DEVICE_NAME}' not found")
    print("Available devices:")
    for d in device_list:
        print(f"  - {d.device_name}")
    import sys
    sys.exit(1)

print(f"\n✓ Found device: {device.device_name}")

# Convert Qiskit circuits to pytket using qbraid and upload to NEXUS
print(f"\nCompiling and uploading {len(circuits)} circuits to NEXUS...")
from qbraid import transpile as qb_transpile
from pytket.passes import DecomposeBoxes

circuit_refs = []
for i, qiskit_circuit in enumerate(circuits):
    # Use qbraid to transpile from qiskit to pytket
    pytket_circuit = qb_transpile(qiskit_circuit, "pytket")

    # Decompose custom gates/boxes to basic gates
    DecomposeBoxes().apply(pytket_circuit)

    # Upload to NEXUS (NEXUS will handle Quantinuum compilation)
    circuit_ref = qnexus.circuits.upload(
        circuit=pytket_circuit,
        project=project,
        name=f"EPLG_circuit_{i}",
    )
    circuit_refs.append(circuit_ref)
    if (i + 1) % 5 == 0 or i == len(circuits) - 1:
        print(f"  Transpiled and uploaded {i+1}/{len(circuits)} circuits")

print(f"✓ All circuits uploaded")

# Compile circuits for Quantinuum backend
print(f"\nCompiling {len(circuit_refs)} circuits for {DEVICE_NAME}...")
compiled_circuit_refs = qnexus.compile(
    programs=circuit_refs,
    backend_config=qnexus.QuantinuumConfig(device_name=DEVICE_NAME),
    name=f"EPLG_Compilation_{DEVICE_NAME}",
    project=project,
    optimisation_level=2,
    timeout=1800,
)
print(f"✓ Circuits compiled")

# Submit the job and wait for results
print(f"\nSubmitting job with {len(compiled_circuit_refs)} circuits, {nshots} shots each...")
print("Waiting for results (this may take a few minutes)...")

results = qnexus.execute(
    programs=compiled_circuit_refs,
    name=f"EPLG_LayerFidelity_{DEVICE_NAME}",
    n_shots=[nshots] * len(compiled_circuit_refs),
    backend_config=qnexus.QuantinuumConfig(device_name=DEVICE_NAME),
    project=project,
    timeout=1800,  # 30 minute timeout
)

print("\n" + "="*60)
print("✓ Job completed!")
print("="*60)
print(f"Received {len(results)} circuit results")

# Process results and run EPLG analysis
from pytket.circuit import BasisOrder
from qiskit.result import Result as QiskitResult
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_experiments.framework import ExperimentData

print("\nProcessing results...")

# Convert Quantinuum results to Qiskit Result format for analysis
experiment_results = []
all_counts_list = []

for i, result in enumerate(results):
    # Get counts with descending lexographic order (MSB first)
    counts = result.get_counts(basis=BasisOrder.dlo)
    # Convert tuple keys to bitstrings
    norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}
    all_counts_list.append(norm_counts)

    # Get the circuit metadata (includes composite_index needed for analysis)
    circuit_metadata = circuits[i].metadata

    # Create ExperimentResult object with metadata
    exp_result = ExperimentResult(
        shots=nshots,
        success=True,
        data=ExperimentResultData(counts=norm_counts),
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
chain_eplgs_quantinuum = [float(x) for x in list(chain_eplgs)]

devices = {DEVICE_NAME: (chain_lens, chain_eplgs_quantinuum)}

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
