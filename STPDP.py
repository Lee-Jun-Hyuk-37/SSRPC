import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, size):
    if size % 2 == 0:
        window = np.ones(size + 1)
        window[0], window[-1] = 0.5, 0.5
        return np.convolve(data, window, 'valid') / size
    else:
        return np.convolve(data, np.ones(size), 'valid') / size


class STPDP:
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length
        self.trend_res = []
        self.detrend_res = []
        self.omega = None

    def omega(self, start_limit=0, end_limit=-1):
        self.trend_res = []
        self.detrend_res = []
        for i in range(1, self.max_length + 1):
            trend = moving_average(self.data, i)
            pow_trend = np.mean((trend - np.mean(trend)) ** 2)
            self.trend_res.append(pow_trend)

            if i == 1:
                detrend = self.data - trend
            else:
                detrend = self.data[i // 2:-(i // 2)] - trend
            pow_detrend = np.mean((detrend - np.mean(detrend)) ** 2)
            self.detrend_res.append(pow_detrend)

        self.trend_res = np.array(self.trend_res)
        self.detrend_res = np.array(self.detrend_res)

        self.omega = (np.where(self.trend_res + self.detrend_res == np.min((self.trend_res + self.detrend_res)[start_limit:end_limit]))[0][0] + 1) * 2
        return self.omega

    def plot(self):
        if self.omega:
            window_size = np.arange(1, self.max_length + 1)
            plt.plot(window_size[:self.max_length + 1], self.trend_res[:self.max_length], label='Trend')
            plt.plot(window_size[:self.max_length + 1], self.detrend_res[:self.max_length], label='Detrended')
            plt.xlabel('Window size')
            plt.ylabel('Power of components')
            plt.legend()
            plt.show()

            plt.plot(window_size[:self.max_length + 1], (self.trend_res + self.detrend_res)[:self.max_length])
            plt.xlabel('Window size')
            plt.ylabel('Power')
            plt.title("Power of Trend + Power of Detrended")
            plt.show()

            print("omega: {}".format(self.omega))

        else:
            print("You need to execute the omega method first")
