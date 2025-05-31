"""
Metriq-gym Platform Introspection Module.

This module provides standardized, device- and provider-agnostic
access to metadata and introspection utilities for quantum computing platforms.

It defines generic functions (using singledispatch) to fetch various metadata or
properties, such as backend versions, coupling graphs, provider details, and job metadata.
Job metadata includes execution time and provider-specific status information such
as queue position when available.

It builds on top of qBraid's runtime module, and in the future,
functions defined here may be upstreamed to qBraid's runtime module.

Submodules:
- device:    introspection functions related to quantum devices,
              such as coupling graphs, backend versions, and calibration data.
- job: standardized queries for job execution details and metadata (e.g. wall-clock execution time).
- provider: functions to introspect provider-specific metadata or capabilities.
"""