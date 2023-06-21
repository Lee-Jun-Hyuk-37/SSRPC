import numpy as np
import matplotlib.pyplot as plt
from nolitsa.nolitsa.lyapunov import mle
from sklearn.linear_model import LinearRegression
from STPDP import STPDP


class SSRPC:
    def __init__(self, data, sample, omega):
        self.data = data
        self.sample = sample
        self.time = None
        self.omega = omega
        self.dim = None
        self.state_space = None
        self.divergence = None

    def reconstruct(self):
        length = len(self.data)
        if length % self.omega == 0:
            pass
        else:
            self.data = self.data[:-(length % self.omega)]

        X = np.zeros(((len(self.data) // self.omega) - 1, self.omega * self.omega))

        for i in range(self.omega):
            tmp = self.data[i:-(self.omega - i)].reshape(-1, self.omega)
            X[:, i * self.omega:i * self.omega + self.omega] = tmp
        X = X.reshape(-1, self.omega)

        S = np.cov(X, rowvar=False)
        D, U = np.linalg.eig(S)
        Z = np.matmul((X - np.tile(np.mean(X, axis=0), [X.shape[0], 1])), U)
        Z = Z.real

        valid_dim = [False for i in range(self.omega)]

        for i in range(self.omega):
            tmp = STPDP(Z[:, i], self.omega).calculate_omega()

            if tmp >= self.omega / 2:
                valid_dim[i] = True
            else:
                break

        valid_dim = np.array(valid_dim)
        self.dim = np.count_nonzero(valid_dim == True)
        self.state_space = Z[:, valid_dim]

        return self.state_space

    def calculate_divergence(self, maxt):
        if self.state_space is not None:
            self.divergence = mle(self.state_space, maxt=maxt, window=int(self.omega * 0.9))
            self.time = np.arange(maxt * self.sample, step=self.sample)
        else:
            print("You need to execute the reconsturct method first")

    def plot_divergence(self, expected=None):
        if self.divergence is not None:
            plt.figure()
            plt.plot(self.time, self.divergence, label="divergence")
            if expected:
                plt.plot(self.time, expected * self.time + self.divergence[0], label="expected")
                plt.legend()
            plt.xlabel("time")
            plt.ylabel('ln<divergence>')
            plt.show()
        else:
            print("You need to execute the divergence method first")

    def mle(self, linear_region):
        start, end = linear_region[0], linear_region[1]
        t = self.time.reshape(-1, 1)
        d = self.divergence.reshape(-1, 1)
        return LinearRegression().fit(t[start:end], d[start:end]).coef_[0, 0]
