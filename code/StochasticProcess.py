import numpy as np
import scipy as sp
from scipy import stats

class StochasticProcess:
    def __init__(self):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        self.mu = np.array([1.0, 2.0])
        self.sigma = np.identity(2)
        self.y_mu = np.array([3.0, 4.0])
        self.y_sigma = np.identity(2)
        self.x_dim = len(self.mu)
        self.y_dim = len(self.y_mu)

    def sample_initial(self):
        return stats.multivariate_normal(self.mu, self.sigma).rvs()

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

    def simulate(self, T=100, seed=12345):
        self.xs= np.zeros(shape=[T, self.x_dim])
        self.ys= np.zeros(shape=[T, self.y_dim])

        # initialise x[0] and y[0]                                           
        self.xs[0] = self.sample_initial()
        self.ys[0] = self.sample_observation(self.xs[0])
        
        for t in range(1,T):
            self.xs[t] = self.sample_transition(self.xs[t-1])
            self.ys[t] = self.sample_observation(self.xs[t])

    def plot(self):
        """ plot each dimension on different windows/lines """
        pylab.figure()
        pylab.plot(self.ys,'k+',label='noisy measurements (ys)')
        pylab.plot(self.xs,'b-',label='hideen state (xs) values')
        pylab.legend()
        pylab.xlabel('Iteration')
        pylab.ylabel('Observation')

