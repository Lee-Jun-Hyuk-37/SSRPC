"""State Space Reconstruction with Principal Components (SSRPC).

Noise-robust state space reconstruction for estimating the maximal
Lyapunov exponent of ergodic scalar time series. See Lee, Park & Ahn
(2023), *Chaos, Solitons & Fractals* 174, 113916.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from . import _patches
from .stpdp import STPDP

_patches.apply_patches()

from nolitsa.nolitsa.lyapunov import mle as _nolitsa_mle  # noqa: E402


ArrayLike = Union[np.ndarray, list, tuple]


def _reconstruct_state_space(data: np.ndarray,
                             omega: int,
                             dimension: Optional[int] = None
                             ) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build the principal-component state space for a time series.

    Returns ``(state_space, dim, data_trimmed)`` where ``state_space``
    has shape ``(N, dim)``, ``dim`` is the number of retained valid
    principal components, and ``data_trimmed`` is the (possibly
    truncated) input used to build the embedding.

    The procedure exactly matches the original ``SSRPC.reconstruct``
    implementation: the input is trimmed so its length is a multiple of
    ``omega``; a Hankel-style matrix of delayed segments is assembled,
    stacked so each row holds ``omega`` consecutive samples; the
    covariance is eigendecomposed and the mean-centred matrix is
    projected onto the eigenvectors; dimensions are kept as long as
    their own STPDP estimate exceeds ``omega / 2`` (starting from the
    first component, stopping at the first failure).
    """
    data = np.asarray(data)
    length = len(data)
    if length % omega != 0:
        data = data[:-(length % omega)]

    X = np.zeros(((len(data) // omega) - 1, omega * omega))
    for i in range(omega):
        tmp = data[i:-(omega - i)].reshape(-1, omega)
        X[:, i * omega:i * omega + omega] = tmp
    X = X.reshape(-1, omega)

    S = np.cov(X, rowvar=False)
    _, U = np.linalg.eig(S)
    Z = np.matmul(
        (X - np.tile(np.mean(X, axis=0), [X.shape[0], 1])), U
    ).real

    valid_dim = np.zeros(omega, dtype=bool)
    if dimension is not None:
        valid_dim[:dimension] = True
    else:
        for i in range(omega):
            tmp = STPDP(Z[:, i], omega).calculate_omega()
            if tmp >= omega / 2:
                valid_dim[i] = True
            else:
                break

    dim = int(np.count_nonzero(valid_dim))
    state_space = Z[:, valid_dim]
    return state_space, dim, data


class SSRPC:
    """State space reconstruction with principal components.

    Parameters
    ----------
    data : array_like
        1-D scalar time series.
    sample : float
        Sampling period (so ``time = arange(maxt) * sample``).
    omega : int
        Mean orbital period. Usually obtained from :class:`STPDP` or
        :func:`ssrpc.estimate_omega`.

    Attributes
    ----------
    state_space : ndarray or None
        Principal-component embedding after :meth:`reconstruct`.
    dim : int or None
        Number of retained principal components.
    time, divergence : ndarray or None
        Output of :meth:`calculate_divergence`.
    """

    def __init__(self, data: ArrayLike, sample: float, omega: int) -> None:
        self.data = np.asarray(data)
        self.sample = float(sample)
        self.omega = int(omega)
        self.time: Optional[np.ndarray] = None
        self.dim: Optional[int] = None
        self.state_space: Optional[np.ndarray] = None
        self.divergence: Optional[np.ndarray] = None

    def reconstruct(self, dimension: Optional[int] = None) -> None:
        """Build the principal-component embedding.

        Parameters
        ----------
        dimension : int, optional
            If given, keep exactly this many leading principal components
            and skip the STPDP-based validity test.
        """
        state_space, dim, data = _reconstruct_state_space(
            self.data, self.omega, dimension=dimension
        )
        self.data = data
        self.dim = dim
        self.state_space = state_space

    def calculate_divergence(self, maxt: int) -> None:
        """Compute average near-neighbour log-divergence up to ``maxt``.

        Uses the Rosenstein et al. (1993) algorithm as implemented in
        ``nolitsa.lyapunov.mle``. The Theiler window is ``int(0.9 *
        omega)``, matching the original SSRPC paper.
        """
        if self.state_space is None:
            raise RuntimeError(
                "reconstruct() must be called before calculate_divergence()."
            )
        self.divergence = _nolitsa_mle(
            self.state_space, maxt=maxt, window=int(self.omega * 0.9)
        )
        self.time = np.arange(maxt * self.sample, step=self.sample)

    def plot_divergence(self, expected: Optional[float] = None) -> None:
        """Plot the divergence curve and (optionally) the expected slope."""
        if self.divergence is None or self.time is None:
            raise RuntimeError(
                "calculate_divergence() must be called before "
                "plot_divergence()."
            )
        plt.figure()
        plt.plot(self.time, self.divergence, label="divergence")
        if expected:
            plt.plot(self.time, expected * self.time + self.divergence[0],
                     label="expected")
            plt.legend()
        plt.xlabel("time")
        plt.ylabel('ln<divergence>')
        plt.show()

    def mle(self, linear_region: Tuple[int, int]) -> float:
        """Maximal Lyapunov exponent as the slope over ``linear_region``.

        ``linear_region`` is a pair of indices into the ``time`` /
        ``divergence`` arrays. A linear regression is fit over
        ``divergence[start:end]`` vs ``time[start:end]`` and the slope
        is returned.
        """
        if self.divergence is None or self.time is None:
            raise RuntimeError(
                "calculate_divergence() must be called before mle()."
            )
        start, end = linear_region
        t = self.time.reshape(-1, 1)
        d = self.divergence.reshape(-1, 1)
        return LinearRegression().fit(t[start:end], d[start:end]).coef_[0, 0]
