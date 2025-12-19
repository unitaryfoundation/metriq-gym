from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
import random
import json
import warnings
import logging
import os


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
    edges = []
    prev_node = None
    for node in path:
        if prev_node is not None:
            if G.has_edge(prev_node, node):
                edges.append((prev_node, node))
            else:
                edges.append((node, prev_node))
        prev_node = node
    return edges
def path_fidelity(path, correct_by_duration: bool = True, readout_scale: float = None):
    """Compute an estimate of the total fidelity of 2-qubit gates on a path.
    If `correct_by_duration` is true, each gate fidelity is worsen by
    scale = max_duration / duration, i.e. gate_fidelity^scale.
    If `readout_scale` > 0 is supplied, readout_fidelity^readout_scale
    for each qubit on the path is multiplied to the total fielity.
    The path is given in node indices form, e.g. [0, 1, 2].
    An external function `to_edges` is used to obtain edge list, e.g. [(0, 1), (1, 2)]."""
    path_edges = to_edges(path)
    max_duration = max(backend.target[twoq_gate][qs].duration for qs in path_edges)

    def gate_fidelity(qpair):
        duration = backend.target[twoq_gate][qpair].duration
        scale = max_duration / duration if correct_by_duration else 1.0
        # 1.25 = (d+1)/d) with d = 4
        return max(0.25, 1 - (1.25 * backend.target[twoq_gate][qpair].error)) ** scale

    def readout_fidelity(qubit):
        return max(0.25, 1 - backend.target["measure"][(qubit,)].error)

    total_fidelity = np.prod([gate_fidelity(qs) for qs in path_edges])
    if readout_scale:
        total_fidelity *= np.prod([readout_fidelity(q) for q in path]) ** readout_scale
    return total_fidelity
def flatten(paths, cutoff=None):  # cutoff not to make run time too large
    return [
        path
        for s, s_paths in paths.items()
        for t, st_paths in s_paths.items()
        for path in st_paths[:cutoff]
        if s < t
    ]

def _pick_twoq_gate(backend, prefer=None):
    if prefer and prefer in getattr(backend.target, "operation_names", []):
        return prefer
    for g in ("ecr", "cx", "cz", "iswap", "rxx", "xx_plus_yy", "xx", "ms"):
        if g in getattr(backend.target, "operation_names", []):
            return g
    return None

def _allowed_edges(G_und, backend=None, twoq_gate=None, require_gate=True):
    """Return a set of undirected edges {(u,v)} allowed for the chain."""
    edges = [tuple(e) for e in G_und.edge_list()]
    if not require_gate or backend is None:
        return {tuple(sorted(e)) for e in edges}
    gate = _pick_twoq_gate(backend, twoq_gate)
    if gate is None:
        return {tuple(sorted(e)) for e in edges}  # can't verify—allow all couplings
    gmap = backend.target[gate]
    #print((gmap))
    allowed = set()
    for u, v in edges:
        if (u, v) in gmap or (v, u) in gmap:
            allowed.add(tuple(sorted((u, v))))
    return allowed

def random_chain_fast(G, length, *, seed=None, backend=None, twoq_gate=None,
                      require_gate=True, restarts=200):
    """
    Sample ONE simple path of `length` nodes quickly.
    Tries `restarts` random seeds; each attempt grows from a random edge,
    extending randomly at head/tail. No global enumeration or deep backtracking.
    """
    rng = random.Random(seed)
    G_und = G.to_undirected(multigraph=False)

    allowed = _allowed_edges(G_und, backend, twoq_gate, require_gate)
    if not allowed:
        raise RuntimeError("No allowed 2-qubit edges found to form a chain.")

    # Build adjacency from allowed edges only
    n_nodes = G_und.num_nodes()
    adj = {i: [] for i in range(n_nodes)}
    for u, v in allowed:
        adj[u].append(v); adj[v].append(u)

    # Keep only nodes that have at least one allowed neighbor
    nodes_with_deg = [n for n in range(n_nodes) if adj[n]]

    # Edge list for random starts
    allowed_edges_list = list(allowed)

    for _ in range(restarts):
        # Start from a random allowed edge: path of length 2 already
        u, v = rng.choice(allowed_edges_list)
        path = [u, v]
        used = {u, v}

        # Grow until target length
        while len(path) < length:
            extended = False
            # Try to extend either end (random order)
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
                # Dead end – restart with a new random edge
                break

        if len(path) == length:
            return path

    raise RuntimeError(
        f"Failed to sample a chain of length={length} after {restarts} restarts. "
        "Try increasing `restarts`, lowering `length`, or setting `require_gate=False`."
    )

# Suppress warnings and errors from qiskit_experiments about deprecated experiment service
warnings.filterwarnings('ignore')
logging.getLogger('qiskit_experiments').setLevel(logging.CRITICAL)
os.environ['QISKIT_EXPERIMENTS_SKIP_SERVICE'] = '1'

QISKIT_IBM_TOKEN="VtlxEceoQEKXDY1peDskgT-pLES3LbEM-dWIm8Ig6U8L"
QISKIT_IBM_CHANNEL="ibm_quantum_platform"
QISKIT_IBM_INSTANCE="crn:v1:bluemix:public:quantum-computing:us-east:a/1d194ddd99a94496b8f253cb72212526:6f07487f-bcc0-435e-a042-3def311b8213::"
service = QiskitRuntimeService(token=QISKIT_IBM_TOKEN, channel=QISKIT_IBM_CHANNEL, instance=QISKIT_IBM_INSTANCE)
backend = service.backend("ibm_fez")
twoq_gate = "cz"
print(f"Device: {backend.name} ({backend.num_qubits} qubits, gate: {twoq_gate})")

num_qubits_in_chain = 120
# Minimal test parameters (comment out for production)
# num_qubits_in_chain = 3
coupling_map = backend.target.build_coupling_map(twoq_gate)
G = coupling_map.graph


CHAIN_TYPE = "best"
if CHAIN_TYPE == "random":
    seed = 12345  # change for different random chains
    qubit_chain = random_chain_fast(
        G,
        num_qubits_in_chain,
        seed=seed,
        backend=backend,      # Qiskit backend object
        twoq_gate=twoq_gate,       # or "ecr"/"cx"/...
        require_gate=False,    # set False to ignore calibration availability
        restarts=100000         # bump if your graph is sparse or the chain is long
    )
    assert len(qubit_chain) == num_qubits_in_chain
    print(f"Qubit chain: {qubit_chain[0]}..{qubit_chain[-1]} ({len(qubit_chain)} qubits)")
    print(f"Predicted layer fidelity: {path_fidelity(qubit_chain):.6f}")

    # decompose the chain into two disjoint layers
    all_pairs = to_edges(qubit_chain)
    two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

elif CHAIN_TYPE == "best":
    # Code for generating the best path
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
    qubit_chain = max(paths, key=path_fidelity)
    # from functools import partial
    # qubit_chain = max(paths, key=partial(path_fidelity, correct_by_duration=True, readout_scale=1.0))
    assert len(qubit_chain) == num_qubits_in_chain
    print(f"Predicted LF from reported 2q-gate EPGs: {path_fidelity(qubit_chain)}")
    print(np.array(qubit_chain))

    # decompose the chain into two disjoint layers
    all_pairs = to_edges(qubit_chain)
    two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

lfexp = LayerFidelity(
    physical_qubits=qubit_chain,
    #physical_qubits=selected_qubit_chain,
    two_qubit_layers=two_disjoint_layers,
    lengths=[2, 4, 8, 16, 30, 50, 70, 100, 150, 200, 300, 500],
    #lengths=[2, 4],                # Regular
    backend=backend,
    num_samples=10,                # Regular
    # lengths=[2],                 # Minimal test
    # num_samples=1,               # Minimal test
    seed=42,
    # USE THIS FOR AWS
    # two_qubit_gate="ecr",
    # one_qubit_basis_gates=["rz", "sx", "x"],
)

# set maximum number of circuits per job to avoid errors due to too large payload
lfexp.experiment_options.max_circuits = 144

# generate all circuits to run
circuits = lfexp.circuits()
print(f"Generated {len(circuits)} circuits")

nshots = 1000    # Regular
# nshots = 10    # Minimal test

# Run the LF experiment (generate circuits and submit the job)
print(f"Submitting {len(circuits)} circuits with {nshots} shots each...")

exp_data = lfexp.run(shots=nshots)
exp_data.auto_save = True
print(f"Run experiment: ID={exp_data.experiment_id} with jobs {exp_data.job_ids}]")
print(exp_data)


with open("data_eplg_notebook_ibm_boston_251208.json", "w") as f:
    json.dump(exp_data.data(), f, indent=4)

df = exp_data.analysis_results(dataframe=True)
print(df)

pfdf = df[(df.name == "ProcessFidelity")]
pfdf[pfdf.value < 0.8]
pfdf[pfdf.quality == "bad"]
pfdf = pfdf.fillna({"value": 0})

# Compute LF by chain length assuming the first layer is full with 2q-gates
lf_sets, lf_qubits = two_disjoint_layers, qubit_chain
full_layer = [None] * (len(lf_sets[0]) + len(lf_sets[1]))
full_layer[::2] = lf_sets[0]
full_layer[1::2] = lf_sets[1]
full_layer = [(lf_qubits[0],)] + full_layer + [(lf_qubits[-1],)]
print(len(full_layer))

assert len(full_layer) == len(lf_qubits) + 1

pfs = [pfdf.loc[pfdf[pfdf.qubits == qubits].index[0], "value"] for qubits in full_layer]
pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
pfs[0] = pfs[0] ** 2
pfs[-1] = pfs[-1] ** 2
print(np.array(pfs))
print(len(pfs))

print(min(pfs))

job = service.job(exp_data.job_ids[0])
JOB_DATE = job.creation_date

# Approximate 1Q RB fidelities at both ends by the square root of 2Q RB fidelity at both ends.
# For example, if we have [(0, 1), (1, 2), (2, 3), (3, 4)] 2Q RB fidelities and if we want to compute a layer fidelity for [1, 2, 3],
# we approximate the 1Q filedities for (1,) and (3,) by the square root of 2Q fidelities of (0, 1) and (3, 4).
chain_lens = list(range(4, len(pfs), 2))

if len(chain_lens) == 0:
    print("\n" + "="*60)
    print("⚠️  Chain too short for EPLG analysis")
    print("="*60)
    print(f"Chain has only {len(pfs)} process fidelities")
    print(f"Need at least 5 for meaningful analysis")
    print(f"\nTo run full analysis, uncomment regular parameters:")
    print(f"  num_qubits_in_chain = 60")
    print(f"  lengths=[2, 4]")
    print(f"  num_samples=10")
    print(f"  nshots = 1000")
    print("="*60)
    import sys
    sys.exit(0)

chain_fids = []
for length in chain_lens:
    w = length + 1  # window size
    fid_w = max(
        np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1])
        for s in range(len(pfs) - w + 1)
    )
    chain_fids.append(fid_w)

num_2q_gates = [length - 1 for length in chain_lens]
chain_eplgs = [
    1 - (fid ** (1 / num_2q)) for num_2q, fid in zip(num_2q_gates, chain_fids)
]
chain_fids_ibm_boston =[float(x) for x in list(chain_fids)]
chain_eplgs_ibm_boston = [float(x) for x in list(chain_eplgs)]
print(chain_lens)
print(chain_fids_ibm_boston)
print(chain_eplgs_ibm_boston)


devices = {"ibm_boston": (chain_lens, chain_eplgs_ibm_boston)}

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
        "picks": picks,  # shows if any target used a nearest neighbor
    })

rows.sort(key=lambda r: r["score_avg"])

hdr = f"{'device':<15} {'EPLG-10':>10} {'EPLG-20':>10} {'EPLG-50':>10} {'EPLG-100':>10} {'score_avg':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['device']:<15} "
          f"{r['EPLG-10']*1e3:>10.6f} {r['EPLG-20']*1e3:>10.6f} {r['EPLG-50']*1e3:>10.6f} {r['EPLG-100']*1e3:>10.6f} "
          f"{r['score_avg']*1e3:>12.6f}")

# If you also want a dict you can reuse programmatically:
eplg_scores = {r["device"]: r["score_avg"] for r in rows}

