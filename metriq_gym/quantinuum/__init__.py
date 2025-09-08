"""Quantinuum provider integration for metriq-gym via qBraid runtime.

This module implements a provider/device/job scaffold targeting Quantinuum
NEXUS targets, including emulators (e.g., H1-1E, H1-2E) and syntax checkers
(e.g., H1-1SC, H1-2SC). It is designed to be extensible to additional
hardware targets as access is provisioned.

Notes:
- Remote submission and validation are implemented via the pytket-quantinuum
  package; you must have a Quantinuum account with appropriate entitlements.
- No local simulator fallback is provided.
"""
