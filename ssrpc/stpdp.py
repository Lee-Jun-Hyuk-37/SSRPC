"""Sum of Trend Power and Detrended Power (STPDP).

Noise-robust estimator of the mean orbital period ``omega`` of an ergodic
scalar time series. See Lee, Park & Ahn (2023), *Chaos, Solitons &
Fractals* 174, 113916.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, list, tuple]


def moving_average(data: ArrayLike, size: int) -> np.ndarray:
    """Centred moving average of window length ``size``.

    For even ``size`` the two edge samples each contribute half a weight,
    so the output preserves the phase centre of the window (hence
    ``len(data) - size + 1`` samples are returned in either case).
    """
    data = np.asarray(data)
    if size % 2 == 0:
        window = np.ones(size + 1)
        window[0], window[-1] = 0.5, 0.5
        return np.convolve(data, window, 'valid') / size
    return np.convolve(data, np.ones(size), 'valid') / size


def estimate_omega(data: ArrayLike, max_length: int,
                   start_limit: int = 0,
                   end_limit: int = -1) -> int:
    """Return the omega estimate without constructing the ``STPDP`` object.

    Parameters
    ----------
    data : array_like
        1-D scalar time series.
    max_length : int
        Maximum window size to sweep when searching for the optimum.
    start_limit, end_limit : int, optional
        Restrict the search to ``(trend + detrended)[start_limit:end_limit]``
        before picking the minimum. Defaults search the full range.

    Returns
    -------
    omega : int
        Estimated mean orbital period.
    """
    return STPDP(data, max_length).calculate_omega(start_limit=start_limit,
                                                   end_limit=end_limit)


class STPDP:
    """Compute omega via the Sum of Trend Power and Detrended Power.

    Parameters
    ----------
    data : array_like
        1-D scalar time series.
    max_length : int
        Maximum window length used in the sweep.

    Attributes
    ----------
    trend_res, detrend_res : ndarray
        Power of the trend and detrended components for each window size
        in ``1 .. max_length``. Populated by :meth:`calculate_omega`.
    omega : int or None
        Estimated mean orbital period. ``None`` before
        :meth:`calculate_omega` has been called.
    """

    def __init__(self, data: ArrayLike, max_length: int) -> None:
        self.data = np.asarray(data)
        self.max_length = int(max_length)
        self.trend_res: np.ndarray = np.empty(0)
        self.detrend_res: np.ndarray = np.empty(0)
        self.omega: Optional[int] = None

    def calculate_omega(self, start_limit: int = 0, end_limit: int = -1) -> int:
        """Run the sweep and return the estimated ``omega``.

        The method is bit-for-bit equivalent to the original implementation:
        window sizes 1..max_length are swept, the (trend + detrended)
        power is minimised over ``[start_limit:end_limit]`` of that
        sequence, and ``omega = (argmin_index + 1) * 2``.
        """
        trend_res = []
        detrend_res = []
        for i in range(1, self.max_length + 1):
            trend = moving_average(self.data, i)
            pow_trend = np.mean((trend - np.mean(trend)) ** 2)
            trend_res.append(pow_trend)

            if i == 1:
                detrend = self.data - trend
            else:
                detrend = self.data[i // 2:-(i // 2)] - trend
            pow_detrend = np.mean((detrend - np.mean(detrend)) ** 2)
            detrend_res.append(pow_detrend)

        self.trend_res = np.asarray(trend_res)
        self.detrend_res = np.asarray(detrend_res)

        stpdp = self.trend_res + self.detrend_res
        idx = int(np.where(stpdp == np.min(stpdp[start_limit:end_limit]))[0][0])
        self.omega = (idx + 1) * 2
        return self.omega

    def plot(self) -> None:
        """Plot the trend/detrend powers and their sum.

        Raises
        ------
        RuntimeError
            If :meth:`calculate_omega` has not been called yet.
        """
        if self.omega is None:
            raise RuntimeError(
                "calculate_omega() must be called before plot()."
            )

        window_size = np.arange(1, self.max_length + 1)
        plt.figure()
        plt.plot(window_size[:self.max_length + 1],
                 self.trend_res[:self.max_length], label='Trend')
        plt.plot(window_size[:self.max_length + 1],
                 self.detrend_res[:self.max_length], label='Detrended')
        plt.xlabel('Window size')
        plt.ylabel('Power of components')
        plt.legend()
        plt.show()

        plt.plot(window_size[:self.max_length + 1],
                 (self.trend_res + self.detrend_res)[:self.max_length],
                 label="STPDP")
        plt.xlabel('Window size')
        plt.ylabel('Power')
        plt.title("Power of Trend + Power of Detrended")
        plt.show()

        print("omega: {}".format(self.omega))
