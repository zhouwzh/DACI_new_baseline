"""Isolated baseline adapters for the DACI simulator.

This package deliberately lives outside ``src``.  It registers additional
schemes only in the process started by ``run_new_baselines.py`` so the existing
DACI source tree and experiment scripts remain unchanged.
"""

from .registry import baseline_metadata, register_supported_schemes, validate_requested_schemes

__all__ = ["baseline_metadata", "register_supported_schemes", "validate_requested_schemes"]
