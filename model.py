import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch
from scipy.stats import norm


class Network:
    def __init__(self, name='rnn', N=128, S=2, R=2, g=1.5, seed=1):
        self.name = name
        npr.seed(seed)
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g = g  # initial recurrent weight scale
        self.S = S  # input
        self.R = R  # readout
        self.sig = 0.01  # noise
        self.z0 = []    # initial condition
        self.ha_before, self.ha, self.ra, self.ua = [], [], [], []  # activity, output
        # initialize weights
        self.ws = (2 * npr.random((N, S)) - 1) / sqrt(S)  # input weights
        self.J = self.g * npr.standard_normal([N, N]) / np.sqrt(N)  # recurrent weights
        self.wr = (2 * npr.random((R, N)) - 1) / sqrt(N)  # readout weights
        self.B = self.wr.T * sqrt(N / R)                  # assume feedback is aligned

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 / (np.cosh(10*np.tanh(x/10)) ** 2) if not torch.is_tensor(x) else 1 / (torch.cosh(10*torch.tanh(x/10)) ** 2)


class Task:
    def __init__(self, name='motor', duration=20, cycles=2, dt=0.1):
        self.name = name
        NT = int(duration / dt)
        # task parameters
        if self.name == 'motor':
            self.cycles, self.T, self.dt, self.NT = cycles, duration, dt, NT
            self.s = 0.0 * np.ones((0, NT))
            self.ustar = (np.sin(2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.75 * np.sin(2 * 2 * pi * np.arange(NT) * cycles / (NT-1)) +
                          0.5 * np.sin(4 * 2 * pi * np.arange(NT) * cycles / (NT-1)))
        elif self.name == 'reach_1d':
            self.T, self.dt, self.NT = duration, dt, NT
            self.s = 0.0 * np.ones((0, NT))
            t1, mu1, sig1 = np.linspace(0, round(NT / 2) - 1, round(NT / 2)), round(NT/4), round(NT/20)
            t2, mu2, sig2 = np.linspace(round(NT/2), NT-1, round(NT/2)), round(3*NT/4), round(NT/20)
            self.ustar = np.concatenate((norm.pdf(t1, loc=mu1, scale=sig1), -norm.pdf(t2, loc=mu2, scale=sig2)))
            self.ustar[self.ustar > 0] /= np.max(self.ustar[self.ustar > 0])
            self.ustar[self.ustar < 0] /= np.max(np.abs(self.ustar[self.ustar < 0]))
            self.vstar = np.cumsum(self.ustar) * dt
            self.xstar = np.cumsum(self.vstar) * dt

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Algorithm:
    def __init__(self, name='reinforce', Nepochs=10000, lr=1e-1, noise=0.1, clip=1e-4):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.lr = lr  # learning rate
        self.noise = noise  # maximum perturbative noise
        self.clip = clip    # clip weight changes above +/-clip
