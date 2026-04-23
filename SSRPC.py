"""Backward-compatible shim for ``from SSRPC import SSRPC``.

The implementation lives in the ``ssrpc`` package; this module re-exports
``SSRPC`` so existing scripts and notebooks keep working. New code should
prefer ``import ssrpc`` / ``from ssrpc import SSRPC``.
"""

from ssrpc import SSRPC

__all__ = ["SSRPC"]
