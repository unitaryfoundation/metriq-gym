"""Verify that all benchmark modules and the registry can be imported successfully.

This test would have caught dataclass field-order violations at class-creation time
before they reached CI, since those errors surface on import rather than at runtime.
"""


def test_registry_imports():
    """Import the top-level registry module, which transitively imports all benchmarks."""
    import metriq_gym.registry  # noqa: F401

    # Also smoke-test direct benchmark imports to isolate which module fails
    import metriq_gym.benchmarks.bseq  # noqa: F401
    import metriq_gym.benchmarks.clops  # noqa: F401
    import metriq_gym.benchmarks.eplg  # noqa: F401
    import metriq_gym.benchmarks.lr_qaoa  # noqa: F401
    import metriq_gym.benchmarks.mirror_circuits  # noqa: F401
    import metriq_gym.benchmarks.qedc_benchmarks  # noqa: F401
    import metriq_gym.benchmarks.qml_kernel  # noqa: F401
    import metriq_gym.benchmarks.wit  # noqa: F401
