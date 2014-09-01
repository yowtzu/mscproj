import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable

class StochasticProcess:
    def __init__(self, config, T=50):
        self.T = T
        self.config = config
        self.simulate()

    def simulate(self):
        self.xs= np.zeros(shape=[self.T, self.config.x_dim])
        self.ys= np.zeros(shape=[self.T, self.config.y_dim])

        # initialise x[0] and y[0]                                           
        self.xs[0] = self.config.sample_initial()
        self.ys[0] = self.config.sample_observation(self.xs[0])
        
        for t in range(1,self.T):
            r = self.config.sample_transition(self.xs[t-1])
            self.xs[t] = r
            y = self.config.sample_observation(self.xs[t])
            self.ys[t] = y

    def plot(self):
        """ plot each dimension on different windows/lines """
        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.config.x_dim)

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

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.config.y_dim)
        
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
