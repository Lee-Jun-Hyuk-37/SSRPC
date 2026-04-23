"""Backward-compatible shim for ``from STPDP import STPDP``.

The implementation lives in the ``ssrpc`` package; this module re-exports
``STPDP`` and ``moving_average`` so existing scripts and notebooks keep
working. New code should prefer ``from ssrpc import STPDP``.
"""

from ssrpc import STPDP, moving_average

__all__ = ["STPDP", "moving_average"]
