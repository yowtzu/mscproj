import scipy as sp
import scipy.stats as sp.stats
import numpy as np

class LinearProcessModel:
    def __init__(self, mu=0, sigma=1, v=0, tau=1):
        self.mu = mu
        self.v = v
        self.sigma = sigma
        self.tau = tau

    def sample_initial(self):
        return sp.stats.norm(self.mu,self.sigma).rvs()

    def initial_density(self, x):
        return sp.stats.norm(self.mu,self.sigma).pdf(x)
 
    def sample_transition(self, x):
        return sp.stats.norm(x+self.mu ,self.sigma).rvs()

    def transition_density(self, x, x_next):
        return sp.stats.norm(x+self.mu, self.sigma).pdf(x_next)
 
    def sample_observation(self, x):
        return sp.stats.norm(x+self.v, self.tau).rvs()
  
    def observation_density(self, y, x):
        return sp.stats.norm(x+self.v, self.tau).pdf(y)

    def simulate(self, T=100, seed=12345):
        np.random.seed(seed)

        self.xs = np.zeros(T)
        self.ys = np.zeros(T)
        
        # initialise
        self.xs[0] = self.sample_initial()
        self.ys[0] = self.sample_observation(self.xs[0])

        for t in range(1,T):
            self.xs[t] = self.sample_transition(self.xs[t-1])
            self.ys[t] = self.sample_observation(self.xs[t])
