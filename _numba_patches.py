"""Backward-compatible shim for the old ``_numba_patches`` module.

Importing this module still applies the nolitsa numba memory-leak patch
as before. The real implementation now lives in ``ssrpc._patches``.
"""

from ssrpc._patches import apply_patches

apply_patches()

__all__ = ["apply_patches"]
