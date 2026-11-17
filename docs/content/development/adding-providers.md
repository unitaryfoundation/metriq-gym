# Adding Provider Support

This guide covers what it takes to run Metriq-Gym benchmarks against a new
quantum provider, and how to decide where that integration should live.

## Decide where the integration belongs

Metriq-Gym executes everything through [qBraid Runtime](https://docs.qbraid.com/runtime/).
A provider therefore reaches Metriq-Gym in one of two ways.

**Upstream it into qBraid.** This is the default, and it is where every cloud
provider has ended up. IBM, IonQ, Braket and Azure were always qBraid-native.
Quantinuum and OriginQ began as custom implementations inside Metriq-Gym and
were both migrated out, in [#742](https://github.com/unitaryfoundation/metriq-gym/pull/742)
and [#798](https://github.com/unitaryfoundation/metriq-gym/pull/798). Upstreaming
means other qBraid users get the provider too, and Metriq-Gym carries no
provider-specific submission code.

**Implement it here.** Only `local`, the Qiskit Aer simulator provider, still
lives in this repository. Implement locally when the provider is not a general
quantum cloud service and would not make sense in qBraid: a simulator, an
in-house device, or a short-lived experiment. Expect it to move upstream if it
turns out to be generally useful.

Either way, the work described in [Teach the qplatform layer](#teach-the-qplatform-layer)
is required, because that part is specific to Metriq-Gym.

## Implementing a provider in Metriq-Gym

Custom providers are registered through qBraid's entry points, so
`load_provider("name")` finds them the same way it finds a qBraid-native one.
Declare them in `pyproject.toml`:

```toml
[project.entry-points."qbraid.providers"]
local = "metriq_gym.local.provider:LocalProvider"

[project.entry-points."qbraid.jobs"]
local = "metriq_gym.local.job:LocalAerJob"
```

Entry points are read from the installed distribution metadata, so after adding
one you need to reinstall before it resolves:

```bash
uv pip install -e . --no-deps
```

See `metriq_gym/local/` for the reference implementation: a `QuantumProvider`
subclass returning `QuantumDevice` objects, and a `QuantumJob` subclass whose
`result()` returns a `Result[GateModelResultData]`.

## Teach the qplatform layer

A provider can load, dispatch and return results and still be unusable, because
benchmarks need facts about the device that qBraid does not expose uniformly.
These live in `metriq_gym/qplatform/`, as `functools.singledispatch` functions
registered per device or job type. This is the step most easily missed, since
nothing fails until a benchmark asks.

In `metriq_gym/qplatform/device.py`:

| Function | Default | Needed for |
|---|---|---|
| `connectivity_graph` | raises `NotImplementedError` | Any topology-aware benchmark (BSEQ, Mirror Circuits, EPLG) |
| `version` | raises `NotImplementedError` | Recording which device revision produced a result |
| `prepare_device_for_dispatch` | no-op | Provider-specific workarounds before submission |

In `metriq_gym/qplatform/job.py`:

| Function | Default | Needed for |
|---|---|---|
| `execution_time` | raises `NotImplementedError` | CLOPS, and runtime recorded on every job |
| `failure_reason` | returns `None` | Reporting why a job failed |

A registration looks like this:

```python
@connectivity_graph.register
def _(device: MyProviderDevice) -> rx.PyGraph:
    return coupling_map_to_graph(device.coupling_map())
```

Return an `rx.PyGraph` (rustworkx). For all-to-all devices,
`rx.generators.complete_graph(num_qubits)` is correct and is what the trapped-ion
and simulator registrations use.

Report the topology the device actually has, not the one it advertises. A device
may report a qubit count far larger than its usable connected component, which
makes a benchmark look runnable when it is not. OriginQ's `WK_C180` reports 180
qubits but exposes 166 with a largest connected component of 127 and a diameter
of 38, so no chain longer than about 39 qubits exists and EPLG at 100 qubits
cannot run on it at all.

## Verify it end to end

Loading and dispatching is not enough to call a provider supported.

```bash
mgym job estimate metriq_gym/schemas/examples/qml_kernel.example.json -p myprovider -d mydevice
mgym job dispatch metriq_gym/schemas/examples/qml_kernel.example.json -p myprovider -d mydevice
mgym job poll <job-id>
```

Check the polled result carries real values. Some providers, including OriginQ's
simulators, return probabilities rather than sampled counts. Metriq-Gym's
benchmarks consume counts, and results carrying no counts are currently dropped
without comment, so the benchmark reports a fabricated `0.0` rather than
failing. See [#799](https://github.com/unitaryfoundation/metriq-gym/issues/799).
A score of exactly zero on a device you expect to work is worth investigating
before you trust anything else the provider returns.

Then run a topology-dependent benchmark, since that exercises the qplatform
registrations that a single-circuit benchmark does not:

```bash
mgym job dispatch metriq_gym/schemas/examples/bseq.example.json -p myprovider -d mydevice
```

## Document it

Add a page under `docs/content/providers/` covering credentials, device naming,
and any provider quirks, then register it in the `nav` block of
`docs/mkdocs.yml`. Add the provider and its credential environment variables to
the tables in `docs/content/providers/overview.md`.
