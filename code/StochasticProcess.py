import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import stats
from collections import Iterable

class StochasticProcess:
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.mu = np.array([0.0])
        self.sigma = np.identity(1)
        self.y_mu = np.array([0.0])
        self.y_sigma = np.identity(1)
        self.x_dim = len(self.mu)
        self.y_dim = len(self.y_mu)

    def sample_initial(self):
        return self.mu
        # return stats.multivariate_normal(self.mu, self.sigma).rvs()

    def initial_density(self, x):
        return stats.multivariate_normal(self.mu, self.sigma).pdf(x)
 
    def sample_transition(self, x):
        return stats.multivariate_normal(x+self.mu, self.sigma).rvs()

    def transition_density(self, x, x_next):
        return stats.multivariate_normal(x+self.mu, self.sigma).pdf(x_next)

    def sample_observation(self, x):
        return stats.multivariate_normal(x+self.y_mu, self.y_sigma).rvs()
  
    def observation_density(self, y, x):
        return stats.multivariate_normal(x+self.y_mu, self.y_sigma).pdf(y)
    
    def simulate(self, T=100):
        self.T = T
        self.xs= np.zeros(shape=[T, self.x_dim])
        self.ys= np.zeros(shape=[T, self.y_dim])

        # initialise x[0] and y[0]                                           
        self.xs[0] = 0
        self.ys[0] = self.sample_observation(self.xs[0])
        
        for t in range(1,T):
            self.xs[t] = self.sample_transition(self.xs[t-1])
            self.ys[t] = self.sample_observation(self.xs[t])

    def plot(self):
        """ plot each dimension on different windows/lines """
        fig, axes = plt.subplots(nrows=1, ncols=self.x_dim)
        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        if not isinstance(axes, Iterable):
            axes = [axes]
            
        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.xs[:,idx], 'r')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')
        fig.tight_layout()

        fig, axes = plt.subplots(nrows=1, ncols=self.y_dim)
        y_min = np.min(self.ys)
        y_max = np.max(self.ys)
        
        if not isinstance(axes, Iterable):
            axes = [axes]
            
        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.ys[:,idx], 'r')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('y')
            ax.set_title('observation states')
        fig.tight_layout()

x = StochasticProcess()
x.simulate()
x.plot()

N_SIM = 10000
res = np.zeros(shape=[N_SIM, 100, 1])
for i in range(N_SIM):
    x = StochasticProcess(100+i)
    x.simulate()
    res[i] = x.xs

x.mean = np.zeros(shape=100)
x.var = np.zeros(shape=100)
for i in range(100):
    x.mean[i] = np.mean(res[:,i, :])
    x.var[i] = np.var(res[:,i,:])
x.mean
