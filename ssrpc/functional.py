"""One-shot convenience wrappers around the SSRPC pipeline.

These functions let callers run the full pipeline in a single expression
without manually stitching :class:`STPDP` and :class:`SSRPC` together.
Results are bit-for-bit identical to the class-based API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression

from .core import SSRPC, _reconstruct_state_space
from .stpdp import STPDP


ArrayLike = Union[np.ndarray, list, tuple]


@dataclass
class SSRPCResult:
    """Bundle of outputs from :func:`estimate_mle`.

    Attributes
    ----------
    omega : int
        Mean orbital period used (either supplied or auto-estimated).
    dim : int
        Number of retained principal components in the embedding.
    state_space : ndarray
        The reconstructed embedding, shape ``(N, dim)``.
    time : ndarray
        Time axis matching ``divergence``.
    divergence : ndarray
        Average log-divergence of nearest-neighbour trajectories.
    mle : float or None
        Slope over ``linear_region`` if one was provided, else ``None``.
    """
    omega: int
    dim: int
    state_space: np.ndarray
    time: np.ndarray
    divergence: np.ndarray
    mle: Optional[float]


def estimate_omega(data: ArrayLike, max_length: int = 500,
                   start_limit: int = 0,
                   end_limit: int = -1) -> int:
    """Estimate the mean orbital period via STPDP.

    Thin wrapper around :meth:`STPDP.calculate_omega` for callers that
    just want the number.
    """
    return STPDP(data, max_length).calculate_omega(
        start_limit=start_limit, end_limit=end_limit
    )


def reconstruct(data: ArrayLike, omega: Optional[int] = None,
                dimension: Optional[int] = None,
                omega_max_length: int = 500,
                omega_start_limit: int = 0
                ) -> Tuple[np.ndarray, int, int]:
    """Reconstruct the principal-component state space.

    Parameters
    ----------
    data : array_like
        1-D scalar time series.
    omega : int, optional
        Mean orbital period. Auto-estimated via STPDP when ``None``.
    dimension : int, optional
        If given, keep the first ``dimension`` components without STPDP
        validity checking (same semantics as ``SSRPC.reconstruct``).
    omega_max_length, omega_start_limit : int, optional
        Forwarded to :func:`estimate_omega` when ``omega`` is ``None``.

    Returns
    -------
    state_space : ndarray
        Embedding of shape ``(N, dim)``.
    dim : int
        Number of retained components.
    omega : int
        The omega used (returned so callers can reuse it).
    """
    if omega is None:
        omega = estimate_omega(data, max_length=omega_max_length,
                               start_limit=omega_start_limit)
    state_space, dim, _ = _reconstruct_state_space(
        np.asarray(data), omega, dimension=dimension
    )
    return state_space, dim, omega


def estimate_mle(data: ArrayLike,
                 sample: float,
                 maxt: int = 500,
                 omega: Optional[int] = None,
                 dimension: Optional[int] = None,
                 linear_region: Optional[Tuple[int, int]] = None,
                 omega_max_length: int = 500,
                 omega_start_limit: int = 0
                 ) -> SSRPCResult:
    """Run the full SSRPC pipeline in one call.

    Parameters
    ----------
    data : array_like
        1-D scalar time series.
    sample : float
        Sampling period of ``data``.
    maxt : int, optional
        Horizon (in samples) for the divergence computation.
    omega : int, optional
        Mean orbital period. Auto-estimated via STPDP when ``None``.
    dimension : int, optional
        If given, skip the STPDP validity test and keep this many
        leading principal components.
    linear_region : tuple of int, optional
        ``(start, end)`` indices into ``divergence`` / ``time`` for the
        linear-fit slope. When ``None`` the MLE is not fit and
        ``result.mle`` is ``None``.
    omega_max_length, omega_start_limit : int, optional
        Forwarded to :func:`estimate_omega` when ``omega`` is ``None``.

    Returns
    -------
    result : SSRPCResult
        Dataclass containing the pieces commonly needed by callers.
    """
    if omega is None:
        omega = estimate_omega(data, max_length=omega_max_length,
                               start_limit=omega_start_limit)

    model = SSRPC(np.asarray(data), sample=sample, omega=omega)
    model.reconstruct(dimension=dimension)
    model.calculate_divergence(maxt=maxt)

    assert model.state_space is not None
    assert model.time is not None
    assert model.divergence is not None
    assert model.dim is not None

    slope: Optional[float] = None
    if linear_region is not None:
        start, end = linear_region
        t = model.time.reshape(-1, 1)
        d = model.divergence.reshape(-1, 1)
        slope = float(
            LinearRegression().fit(t[start:end], d[start:end]).coef_[0, 0]
        )

    return SSRPCResult(
        omega=omega,
        dim=model.dim,
        state_space=model.state_space,
        time=model.time,
        divergence=model.divergence,
        mle=slope,
    )
