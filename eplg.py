"""Unified EPLG benchmark script for IBM, AWS, and Quantinuum providers.

Usage:
    Edit the TUNABLE PARAMETERS section below, then run:
    python eplg.py

Note: The EPLG table shows scores at 10, 20, 50, and 100 qubits, computed automatically.
"""
import os
import random
import numpy as np
import rustworkx as rx
from dotenv import load_dotenv
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity

# Load environment variables
load_dotenv()


# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================

# --- Provider and device selection ---
PROVIDER = "quantinuum"  # "ibm", "aws", or "quantinuum"
DEVICE = "H1-1LE"  # IBM: device name, AWS: "local", Quantinuum: "H1-1LE", "H1-1", etc.

# --- Common parameters ---
LENGTHS = [2, 4]  # Circuit depths to test
NUM_SAMPLES = 3  # Number of random circuits per depth
SHOTS = 100  # Measurement shots per circuit
SEED = 12345  # Random seed for reproducibility

# --- IBM-specific parameters ---
IBM_NUM_QUBITS_IN_CHAIN = 120
IBM_CHAIN_TYPE = "best"  # "random" or "best" - "best" searches all paths for highest fidelity
IBM_TWOQ_GATE = None  # None = auto-detect. Options: "ecr", "cx", "cz", etc.

# --- AWS-specific parameters ---
AWS_NUM_QUBITS_IN_CHAIN = 10
AWS_TWOQ_GATE = "cz"
AWS_ONE_QUBIT_BASIS_GATES = ["rz", "rx", "x"]
AWS_NOISE_PROB = 0.01  # Depolarizing noise probability (1% = 0.01)

# --- Quantinuum-specific parameters ---
QUANTINUUM_NUM_QUBITS_IN_CHAIN = 10
QUANTINUUM_TWOQ_GATE = "cz"
QUANTINUUM_ONE_QUBIT_BASIS_GATES = ["rz", "rx", "x"]
QUANTINUUM_NUM_QUBITS = 20  # Total qubits available on H1 series devices
QUANTINUUM_OPT_LEVEL = 1  # Compilation optimization level (0-3)


# ============================================================================
# Common Utility Functions
# ============================================================================

def eplg_score_at_lengths(chain_lens, chain_eplgs, targets=[10, 20, 50, 100]):
    """Compute EPLG score at specific qubit lengths."""
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
    """Create a complete graph (all-to-all connectivity) for trapped-ion and IQM devices."""
    G = rx.PyDiGraph()
    for i in range(num_qubits):
        G.add_node(i)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            G.add_edge(i, j, None)
            G.add_edge(j, i, None)
    return G


def random_chain_complete_graph(num_qubits, length, seed=None):
    """Sample a random chain from a complete graph."""
    rng = random.Random(seed)
    if length > num_qubits:
        raise ValueError(f"Chain length {length} cannot exceed number of qubits {num_qubits}")
    qubit_chain = rng.sample(range(num_qubits), length)
    return qubit_chain


# ============================================================================
# IBM-specific Functions
# ============================================================================

def _pick_twoq_gate(backend, prefer=None):
    """Pick a two-qubit gate from the backend."""
    if prefer and prefer in getattr(backend.target, "operation_names", []):
        return prefer
    for g in ("ecr", "cx", "cz", "iswap", "rxx", "xx_plus_yy", "xx", "ms"):
        if g in getattr(backend.target, "operation_names", []):
            return g
    return None


def _allowed_edges(G_und, backend=None, twoq_gate=None, require_gate=True):
    """Return a set of undirected edges allowed for the chain."""
    edges = [tuple(e) for e in G_und.edge_list()]
    if not require_gate or backend is None:
        return {tuple(sorted(e)) for e in edges}
    gate = _pick_twoq_gate(backend, twoq_gate)
    if gate is None:
        return {tuple(sorted(e)) for e in edges}
    gmap = backend.target[gate]
    allowed = set()
    for u, v in edges:
        if (u, v) in gmap or (v, u) in gmap:
            allowed.add(tuple(sorted((u, v))))
    return allowed


def random_chain_fast(G, length, *, seed=None, backend=None, twoq_gate=None,
                      require_gate=True, restarts=200):
    """Sample ONE simple path of length nodes quickly."""
    rng = random.Random(seed)
    G_und = G.to_undirected(multigraph=False)

    allowed = _allowed_edges(G_und, backend, twoq_gate, require_gate)
    if not allowed:
        raise RuntimeError("No allowed 2-qubit edges found to form a chain.")

    n_nodes = G_und.num_nodes()
    adj = {i: [] for i in range(n_nodes)}
    for u, v in allowed:
        adj[u].append(v)
        adj[v].append(u)

    allowed_edges_list = list(allowed)

    for _ in range(restarts):
        u, v = rng.choice(allowed_edges_list)
        path = [u, v]
        used = {u, v}

        while len(path) < length:
            extended = False
            for side in rng.sample(("head", "tail"), 2):
                if side == "head":
                    cur = path[0]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.insert(0, w)
                        used.add(w)
                        extended = True
                        break
                else:
                    cur = path[-1]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.append(w)
                        used.add(w)
                        extended = True
                        break
            if not extended:
                break

        if len(path) == length:
            return path

    raise RuntimeError(
        f"Failed to sample a chain of length={length} after {restarts} restarts. "
        "Try increasing `restarts`, lowering `length`, or setting `require_gate=False`."
    )


def path_fidelity(path, backend, twoq_gate, correct_by_duration=True, readout_scale=None):
    """Compute an estimate of the total fidelity of 2-qubit gates on a path."""
    def to_edges_ibm(path):
        edges = []
        prev_node = None
        for node in path:
            if prev_node is not None:
                G = backend.target.build_coupling_map(twoq_gate).graph
                if G.has_edge(prev_node, node):
                    edges.append((prev_node, node))
                else:
                    edges.append((node, prev_node))
            prev_node = node
        return edges

    path_edges = to_edges_ibm(path)
    max_duration = max(backend.target[twoq_gate][qs].duration for qs in path_edges)

    def gate_fidelity(qpair):
        duration = backend.target[twoq_gate][qpair].duration
        scale = max_duration / duration if correct_by_duration else 1.0
        return max(0.25, 1 - (1.25 * backend.target[twoq_gate][qpair].error)) ** scale

    def readout_fidelity(qubit):
        return max(0.25, 1 - backend.target["measure"][(qubit,)].error)

    total_fidelity = np.prod([gate_fidelity(qs) for qs in path_edges])
    if readout_scale:
        total_fidelity *= np.prod([readout_fidelity(q) for q in path]) ** readout_scale
    return total_fidelity


def flatten(paths, cutoff=None):
    """Flatten paths dictionary."""
    return [
        path
        for s, s_paths in paths.items()
        for t, st_paths in s_paths.items()
        for path in st_paths[:cutoff]
        if s < t
    ]


def setup_ibm_provider(device_name, twoq_gate=IBM_TWOQ_GATE):
    """Setup IBM Quantum provider and backend."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Load credentials from environment
    token = os.getenv("QISKIT_IBM_TOKEN")
    channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")
    instance = os.getenv("QISKIT_IBM_INSTANCE")

    if not token:
        raise ValueError("QISKIT_IBM_TOKEN not found in environment variables. Please set it in .env file.")

    # Initialize service
    service_kwargs = {"token": token, "channel": channel}
    if instance:
        service_kwargs["instance"] = instance

    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(device_name)

    # Auto-detect two-qubit gate if not specified
    if twoq_gate is None:
        twoq_gate = _pick_twoq_gate(backend)

    print(f"Device: {backend.name} ({backend.num_qubits} qubits, gate: {twoq_gate})")

    return backend, twoq_gate, service


def generate_ibm_chain(backend, twoq_gate, num_qubits_in_chain, chain_type="best", seed=12345):
    """Generate qubit chain for IBM device."""
    coupling_map = backend.target.build_coupling_map(twoq_gate)
    G = coupling_map.graph

    if chain_type == "random":
        qubit_chain = random_chain_fast(
            G,
            num_qubits_in_chain,
            seed=seed,
            backend=backend,
            twoq_gate=twoq_gate,
            require_gate=False,
            restarts=100000
        )
        print(f"Qubit chain: {qubit_chain[0]}..{qubit_chain[-1]} ({len(qubit_chain)} qubits)")
        print(f"Predicted layer fidelity: {path_fidelity(qubit_chain, backend, twoq_gate):.6f}")

    elif chain_type == "best":
        paths = rx.all_pairs_all_simple_paths(
            G.to_undirected(multigraph=False),
            min_depth=num_qubits_in_chain,
            cutoff=num_qubits_in_chain,
        )
        paths = flatten(paths, cutoff=400)
        if not paths:
            raise Exception(
                f"No qubit chain with length={num_qubits_in_chain} exists in {backend.name}. Try smaller num_qubits_in_chain."
            )

        print(f"Selecting the best from {len(paths)} candidate paths (will take a few minutes)")
        qubit_chain = max(paths, key=lambda p: path_fidelity(p, backend, twoq_gate))
        print(f"Predicted LF from reported 2q-gate EPGs: {path_fidelity(qubit_chain, backend, twoq_gate)}")
        print(np.array(qubit_chain))

    return qubit_chain


def run_ibm_experiment(backend, twoq_gate, qubit_chain, two_disjoint_layers, lengths, num_samples, nshots):
    """Run LayerFidelity experiment on IBM backend."""
    lfexp = LayerFidelity(
        physical_qubits=qubit_chain,
        two_qubit_layers=two_disjoint_layers,
        lengths=lengths,
        backend=backend,
        num_samples=num_samples,
        seed=42,
    )

    lfexp.experiment_options.max_circuits = 2 * num_samples * len(lengths)
    circuits = lfexp.circuits()
    print(f"Generated {len(circuits)} circuits")
    print(f"Submitting {len(circuits)} circuits with {nshots} shots each...")

    exp_data = lfexp.run(shots=nshots)
    exp_data.auto_save = True
    print(f"Run experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}")
    print(exp_data)

    return exp_data


# ============================================================================
# AWS-specific Functions
# ============================================================================

def setup_aws_provider():
    """Setup AWS Braket local simulator with noise."""
    from braket.devices import LocalSimulator

    device = LocalSimulator("braket_dm")
    device_name = "LocalSimulator (DM, noisy)"
    num_qubits = 10

    print(f"Device: {device_name}")
    print(f"  Qubits: {num_qubits}")
    print("  Topology: Complete graph (all-to-all connectivity)")
    print("  Noise: Enabled (1% depolarizing noise)")

    return device, device_name, num_qubits


def run_aws_experiment(device, device_name, qubit_chain, two_disjoint_layers, lengths, num_samples, nshots):
    """Run LayerFidelity experiment on AWS Braket local simulator with noise."""
    from qbraid import transpile as qb_transpile
    from braket.circuits import Circuit

    twoq_gate = AWS_TWOQ_GATE
    one_qubit_basis_gates = AWS_ONE_QUBIT_BASIS_GATES

    lfexp = LayerFidelity(
        physical_qubits=qubit_chain,
        two_qubit_layers=two_disjoint_layers,
        lengths=lengths,
        num_samples=num_samples,
        seed=42,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=one_qubit_basis_gates,
    )

    lfexp.experiment_options.max_circuits = 2 * num_samples * len(lengths)
    circuits = lfexp.circuits()
    print(f"\nGenerated {len(circuits)} circuits")

    # Convert circuits to Braket format
    print(f"\nConverting {len(circuits)} circuits to Braket format...")
    braket_circuits = []
    for i, qiskit_circuit in enumerate(circuits):
        braket_circuit = qb_transpile(qiskit_circuit, "braket")
        braket_circuits.append(braket_circuit)
        if (i + 1) % 5 == 0 or i == len(circuits) - 1:
            print(f"  Converted {i+1}/{len(circuits)} circuits")

    # Add noise model
    print("\nAdding noise model to circuits...")
    noise_prob = AWS_NOISE_PROB
    noisy_circuits = []
    for original_circuit in braket_circuits:
        gate_instructions = []
        measurement_instructions = []

        for instruction in original_circuit.instructions:
            if instruction.operator.name == 'Measure':
                measurement_instructions.append(instruction)
            else:
                gate_instructions.append(instruction)

        noisy_circuit = Circuit()
        for instruction in gate_instructions:
            noisy_circuit.add_instruction(instruction)
            target_qubits = list(instruction.target)
            for qubit in target_qubits:
                noisy_circuit.depolarizing(qubit, probability=noise_prob)

        for instruction in measurement_instructions:
            noisy_circuit.add_instruction(instruction)

        noisy_circuits.append(noisy_circuit)

    braket_circuits = noisy_circuits
    print(f"Added {noise_prob*100}% depolarizing noise after all gates")

    # Execute circuits
    print(f"\nExecuting {len(braket_circuits)} circuits with {nshots} shots each...")
    results = []
    for i, circuit in enumerate(braket_circuits):
        result = device.run(circuit, shots=nshots).result()
        results.append(result)
        if (i + 1) % 5 == 0 or i == len(braket_circuits) - 1:
            print(f"  Executed {i+1}/{len(braket_circuits)} circuits")

    print("\n" + "="*60)
    print("Job completed!")
    print("="*60)

    return convert_braket_results(results, circuits, device_name, lfexp, nshots)


def convert_braket_results(results, circuits, device_name, lfexp, nshots):
    """Convert Braket results to Qiskit format."""
    from qiskit.result import Result as QiskitResult
    from qiskit.result.models import ExperimentResult, ExperimentResultData
    from qiskit_experiments.framework import ExperimentData

    experiment_results = []
    for i, result in enumerate(results):
        measurements = result.measurements
        counts_dict = {}
        for measurement in measurements:
            bitstring = "".join(str(int(b)) for b in measurement)
            counts_dict[bitstring] = counts_dict.get(bitstring, 0) + 1

        circuit_metadata = circuits[i].metadata
        exp_result = ExperimentResult(
            shots=nshots,
            success=True,
            data=ExperimentResultData(counts=counts_dict),
            header={"name": f"circuit_{i}", "metadata": circuit_metadata}
        )
        experiment_results.append(exp_result)

    qiskit_result = QiskitResult(
        backend_name=device_name,
        backend_version="1.0",
        job_id="eplg_job",
        success=True,
        results=experiment_results
    )

    exp_data = ExperimentData(experiment=lfexp)
    exp_data.add_data(qiskit_result)
    exp_data.auto_save = False

    return exp_data


# ============================================================================
# Quantinuum-specific Functions
# ============================================================================

def setup_quantinuum_provider(device_name="H1-1LE"):
    """Setup Quantinuum NEXUS provider."""
    import qnexus

    # Get the project
    projects = list(qnexus.projects.get_all())
    if not projects:
        raise RuntimeError("No projects found. Create one at https://nexus.quantinuum.com/")

    project_name = os.getenv("QUANTINUUM_NEXUS_PROJECT_NAME", "metriq-gym")
    project = None
    for p in projects:
        p_name = p.annotations.name if hasattr(p.annotations, 'name') else None
        if p_name == project_name:
            project = p
            break

    if project is None:
        raise RuntimeError(f"Project '{project_name}' not found")

    print(f"Using project: {project_name} ({project.id})")

    # Get device
    all_devices = qnexus.devices.get_all()
    device_list = list(all_devices)

    device = None
    for d in device_list:
        if d.device_name == device_name:
            device = d
            break

    if device is None:
        raise RuntimeError(f"Device '{device_name}' not found")

    print(f"Device: {device.device_name}")
    print("  Topology: Complete graph (all-to-all connectivity)")

    return device, project


def run_quantinuum_experiment(device, project, qubit_chain, two_disjoint_layers, lengths, num_samples, nshots):
    """Run LayerFidelity experiment on Quantinuum."""
    import qnexus
    from qbraid import transpile as qb_transpile
    from pytket.passes import DecomposeBoxes

    twoq_gate = QUANTINUUM_TWOQ_GATE
    one_qubit_basis_gates = QUANTINUUM_ONE_QUBIT_BASIS_GATES

    lfexp = LayerFidelity(
        physical_qubits=qubit_chain,
        two_qubit_layers=two_disjoint_layers,
        lengths=lengths,
        num_samples=num_samples,
        seed=42,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=one_qubit_basis_gates,
    )

    lfexp.experiment_options.max_circuits = 2 * num_samples * len(lengths)
    circuits = lfexp.circuits()
    print(f"\nGenerated {len(circuits)} circuits")

    # Convert to pytket and upload
    print(f"\nCompiling and uploading {len(circuits)} circuits to NEXUS...")
    circuit_refs = []
    for i, qiskit_circuit in enumerate(circuits):
        pytket_circuit = qb_transpile(qiskit_circuit, "pytket")
        DecomposeBoxes().apply(pytket_circuit)

        circuit_ref = qnexus.circuits.upload(
            circuit=pytket_circuit,
            project=project,
            name=f"EPLG_circuit_{i}",
        )
        circuit_refs.append(circuit_ref)
        if (i + 1) % 5 == 0 or i == len(circuits) - 1:
            print(f"  Transpiled and uploaded {i+1}/{len(circuits)} circuits")

    print("All circuits uploaded")

    # Compile
    opt_level = QUANTINUUM_OPT_LEVEL
    print(f"\nCompiling {len(circuit_refs)} circuits for {device.device_name}...")
    compiled_circuit_refs = qnexus.compile(
        programs=circuit_refs,
        backend_config=qnexus.QuantinuumConfig(device_name=device.device_name),
        name=f"EPLG_Compilation_{device.device_name}",
        project=project,
        optimisation_level=opt_level,
        timeout=1800,
    )
    print("Circuits compiled")

    # Execute
    print(f"\nSubmitting job with {len(compiled_circuit_refs)} circuits, {nshots} shots each...")
    results = qnexus.execute(
        programs=compiled_circuit_refs,
        name=f"EPLG_LayerFidelity_{device.device_name}",
        n_shots=[nshots] * len(compiled_circuit_refs),
        backend_config=qnexus.QuantinuumConfig(device_name=device.device_name),
        project=project,
        timeout=1800,
    )

    print("\n" + "="*60)
    print("Job completed!")
    print("="*60)

    return convert_quantinuum_results(results, circuits, device.device_name, lfexp, nshots)


def convert_quantinuum_results(results, circuits, device_name, lfexp, nshots):
    """Convert Quantinuum results to Qiskit format."""
    from pytket.circuit import BasisOrder
    from qiskit.result import Result as QiskitResult
    from qiskit.result.models import ExperimentResult, ExperimentResultData
    from qiskit_experiments.framework import ExperimentData

    experiment_results = []
    for i, result in enumerate(results):
        counts = result.get_counts(basis=BasisOrder.dlo)
        norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}

        circuit_metadata = circuits[i].metadata
        exp_result = ExperimentResult(
            shots=nshots,
            success=True,
            data=ExperimentResultData(counts=norm_counts),
            header={"name": f"circuit_{i}", "metadata": circuit_metadata}
        )
        experiment_results.append(exp_result)

    qiskit_result = QiskitResult(
        backend_name=device_name,
        backend_version="1.0",
        job_id="eplg_job",
        success=True,
        results=experiment_results
    )

    exp_data = ExperimentData(experiment=lfexp)
    exp_data.add_data(qiskit_result)
    exp_data.auto_save = False

    return exp_data


# ============================================================================
# Common Analysis Functions
# ============================================================================

def analyze_eplg_results(exp_data, two_disjoint_layers, qubit_chain):
    """Run EPLG analysis on experiment data."""
    import time

    print("Running EPLG analysis...")
    exp_data.block_for_results()

    # Run analysis
    exp_data.experiment.analysis.run(exp_data)

    # Wait for analysis to complete
    max_wait = 60
    waited = 0
    status = str(exp_data.analysis_status())

    while "RUNNING" in status and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5
        status = str(exp_data.analysis_status())
        if int(waited) != int(waited - 0.5):
            print(f"Waiting for analysis... {waited:.0f}s")

    if exp_data.analysis_errors():
        print(f"Analysis errors: {exp_data.analysis_errors()}")

    # Get results
    df = exp_data.analysis_results(dataframe=True)
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
        print("Chain too short for EPLG analysis")
        print("="*60)
        print(f"Chain has only {len(pfs)} process fidelities")
        print("Need at least 5 for meaningful analysis")
        return None, None

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

    return chain_lens, [float(x) for x in chain_eplgs]


def print_eplg_table(device_name, chain_lens, chain_eplgs):
    """Print EPLG results table."""
    devices = {device_name: (chain_lens, chain_eplgs)}

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


# ============================================================================
# Main Function
# ============================================================================

def main():
    # Set chain lengths and chain type based on provider
    if PROVIDER == "ibm":
        num_qubits_in_chain = IBM_NUM_QUBITS_IN_CHAIN
        chain_type = IBM_CHAIN_TYPE
    elif PROVIDER == "aws":
        num_qubits_in_chain = AWS_NUM_QUBITS_IN_CHAIN
    elif PROVIDER == "quantinuum":
        num_qubits_in_chain = QUANTINUUM_NUM_QUBITS_IN_CHAIN
    else:
        raise ValueError(f"Invalid PROVIDER: {PROVIDER}. Must be 'ibm', 'aws', or 'quantinuum'")

    print("="*60)
    print(f"EPLG Benchmark - {PROVIDER.upper()}")
    print("="*60)

    # Provider-specific execution
    if PROVIDER == "ibm":
        backend, twoq_gate, service = setup_ibm_provider(DEVICE)
        qubit_chain = generate_ibm_chain(
            backend, twoq_gate, num_qubits_in_chain, chain_type, SEED
        )
        all_pairs = to_edges(qubit_chain)
        two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

        exp_data = run_ibm_experiment(
            backend, twoq_gate, qubit_chain, two_disjoint_layers,
            LENGTHS, NUM_SAMPLES, SHOTS
        )

        device_name = backend.name

    elif PROVIDER == "aws":
        device, device_name, num_qubits = setup_aws_provider()
        qubit_chain = random_chain_complete_graph(num_qubits, num_qubits_in_chain, SEED)
        print(f"Qubit chain: {qubit_chain}")

        all_pairs = to_edges(qubit_chain)
        two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

        exp_data = run_aws_experiment(
            device, device_name, qubit_chain, two_disjoint_layers,
            LENGTHS, NUM_SAMPLES, SHOTS
        )

    elif PROVIDER == "quantinuum":
        device, project = setup_quantinuum_provider(DEVICE)
        num_qubits = QUANTINUUM_NUM_QUBITS
        qubit_chain = random_chain_complete_graph(num_qubits, num_qubits_in_chain, SEED)
        print(f"Qubit chain: {qubit_chain}")

        all_pairs = to_edges(qubit_chain)
        two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

        exp_data = run_quantinuum_experiment(
            device, project, qubit_chain, two_disjoint_layers,
            LENGTHS, NUM_SAMPLES, SHOTS
        )
        device_name = device.device_name

    # Analyze results
    chain_lens, chain_eplgs = analyze_eplg_results(exp_data, two_disjoint_layers, qubit_chain)

    if chain_lens is not None:
        print_eplg_table(device_name, chain_lens, chain_eplgs)


if __name__ == "__main__":
    main()
