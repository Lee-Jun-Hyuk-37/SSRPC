"""SSRPC: state space reconstruction with principal components.

Reference
---------
Lee, J. H., Park, I. S., & Ahn, J. (2023). "Noise-robust estimation of
the maximal Lyapunov exponent based on state space reconstruction with
principal components." *Chaos, Solitons & Fractals* 174, 113916.
https://doi.org/10.1016/j.chaos.2023.113916

Quick start
-----------
>>> import ssrpc
>>> result = ssrpc.estimate_mle(data, sample=0.01, maxt=500,
...                             linear_region=(0, 200))
>>> result.mle
0.149...

Or the class-based API:

>>> omega = ssrpc.estimate_omega(data, max_length=500)
>>> model = ssrpc.SSRPC(data, sample=0.01, omega=omega)
>>> model.reconstruct()
>>> model.calculate_divergence(maxt=500)
>>> model.mle((0, 200))

Importing this package applies the numba memory-leak patch to
``nolitsa.utils`` automatically; the patch is idempotent.
"""

from . import _patches as _patches_module

_patches_module.apply_patches()

from .stpdp import STPDP, moving_average  # noqa: E402
from .core import SSRPC  # noqa: E402
from .functional import (  # noqa: E402
    SSRPCResult,
    estimate_mle,
    estimate_omega,
    reconstruct,
)

__all__ = [
    "STPDP",
    "SSRPC",
    "SSRPCResult",
    "estimate_mle",
    "estimate_omega",
    "reconstruct",
    "moving_average",
]

__version__ = "0.2.0"
