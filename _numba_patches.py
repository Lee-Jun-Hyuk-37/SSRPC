"""Replace nolitsa's jit-compiled distance functions with loop-based versions.

The upstream implementations in ``nolitsa.utils`` use the pattern
``np.array(list(map(np.sum, ...)))`` inside a ``@jit(nopython=True)`` body.
Under numba, that pattern materialises a temporary reflected/typed list on
every call and leaks a steady amount of heap memory that is never reclaimed
(hundreds of MB over a few thousand calls when the input has thousands of
rows).  ``lyapunov.mle`` invokes ``utils.dist`` inside an inner loop, so a
single MLE estimation over long series triggers a visible leak.

The replacements below compute exactly the same quantities (L1, L2 and
L-infinity row norms) with explicit loops writing into a pre-allocated
output buffer.  They are bit-for-bit equivalent to the originals for
finite inputs and do not leak.
"""

import numpy as np
from numba import jit

from nolitsa.nolitsa import utils as _nolitsa_utils


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True)
def _cityblock_dist(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            s += diff if diff >= 0.0 else -diff
        out[i] = s
    return out


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True)
def _euclidean_dist(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            s += diff * diff
        out[i] = np.sqrt(s)
    return out


@jit("float64[:](float64[:, :], float64[:, :])", nopython=True)
def _chebyshev_dist(x, y):
    n = x.shape[0]
    d = x.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        m = 0.0
        for j in range(d):
            diff = x[i, j] - y[i, j]
            a = diff if diff >= 0.0 else -diff
            if a > m:
                m = a
        out[i] = m
    return out


_PATCHED_FLAG = "_ssrpc_distance_patch_applied"


def apply_patches():
    """Swap the leaky distance functions in ``nolitsa.utils`` in place.

    ``utils.dist`` looks up ``cityblock_dist``/``euclidean_dist``/
    ``chebyshev_dist`` from the ``utils`` module globals on every call, so
    rebinding the module attributes here is enough to redirect all future
    calls through the non-leaking loop-based implementations.
    """
    if getattr(_nolitsa_utils, _PATCHED_FLAG, False):
        return

    _nolitsa_utils.cityblock_dist = _cityblock_dist
    _nolitsa_utils.euclidean_dist = _euclidean_dist
    _nolitsa_utils.chebyshev_dist = _chebyshev_dist
    setattr(_nolitsa_utils, _PATCHED_FLAG, True)


apply_patches()
