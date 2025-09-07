"""Quantinuum provider integration for metriq-gym via qBraid runtime.

This module implements a provider/device/job scaffold targeting
Quantinuum NEXUS emulator devices (e.g., H1-1E, H1-2E).

Notes:
- Remote submission is designed to use the pytket-quantinuum package.
- No local simulator fallback is provided; a Quantinuum account and
  credentials are required.
"""
