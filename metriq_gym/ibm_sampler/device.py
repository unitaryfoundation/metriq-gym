"""IBM device with session, twirling, and parameterized circuit support.

This module defines ``IBMSamplerDevice``, a subclass of qBraid's
``QiskitBackend`` that overrides ``submit()`` to optionally run inside
an IBM Runtime Session and apply twirling.  It accepts SamplerV2 PUBs
directly, matching the ``SamplerV2.run()`` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from qiskit.primitives.containers.sampler_pub import SamplerPubLike
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime.options import TwirlingOptions

from qbraid.runtime.ibm.device import QiskitBackend
from qbraid.runtime.ibm.job import QiskitJob

if TYPE_CHECKING:
    import qiskit_ibm_runtime

    import qbraid.runtime


class IBMSamplerDevice(QiskitBackend):
    """IBM device that submits jobs via SamplerV2 with optional session and twirling.

    Inherits all metadata, transpilation, connectivity, and status logic
    from ``QiskitBackend``.  Only ``submit()`` is overridden.
    """

    def __init__(
        self,
        profile: qbraid.runtime.TargetProfile,
        service: Optional[qiskit_ibm_runtime.QiskitRuntimeService] = None,
    ):
        super().__init__(profile=profile, service=service)

    # pylint: disable-next=arguments-differ
    def submit(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None,
        twirling_options: TwirlingOptions | None = None,
        use_session: bool = True,
    ) -> QiskitJob:
        """Submit PUBs via SamplerV2 with optional session and twirling.

        Mirrors the ``SamplerV2.run()`` interface.  Each PUB can be a bare
        ``QuantumCircuit`` or a tuple of ``(circuit,)``,
        ``(circuit, parameter_values)``, or
        ``(circuit, parameter_values, shots)``.

        See: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2#run

        Args:
            pubs: An iterable of SamplerV2 Primitive Unified Blocs.
            shots: Number of shots per PUB.  If *None* the sampler default
                is used.  Per-PUB shots in the PUB tuple take precedence.
            twirling_options: Optional ``TwirlingOptions`` forwarded to the
                ``SamplerOptions``.  Useful for gate-twirled CLOPS runs.
            use_session: If *True* (default), submit inside an IBM Runtime
                Session for lower latency on iterative workloads.  Set to
                *False* to use a plain backend, which avoids session
                overhead and works on plans that do not support sessions.

        Returns:
            A ``QiskitJob`` wrapping the underlying IBM Runtime job.
        """
        options = SamplerOptions(
            experimental={"execution": {"fast_parametric_update": True}},
        )
        if twirling_options is not None:
            options.twirling = twirling_options

        if use_session:
            with Session(backend=self._backend) as session:
                sampler = Sampler(mode=session, options=options)
                job = sampler.run(pubs, shots=shots)
        else:
            sampler = Sampler(mode=self._backend, options=options)
            job = sampler.run(pubs, shots=shots)

        return QiskitJob(job.job_id(), job=job, device=self)
