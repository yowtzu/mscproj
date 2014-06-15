import scipy as sp
import numpy as np

class NonLinearProcessModel:
    def __init__(self, sigma_0=5, sigma_v=10, sigma_w=1):
        self.sigma_0 = sigma_0
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w

    def sample_initial(self):
        return sp.stats.norm(0,self.sigma_0).rvs()

    def initial_density(self, x):
        return self.initial_distribution.pdf(x)
 
    def sample_transition(self, t, x):
        return x/2 + 25*x/(1+x**2) + self.stats.norm(0,self.sigma_v).rvs()

    def transition_density(self, t, x, x_next):
        return sp.stats.norm(x/2 + 25*x/(1+x**2),self.sigma_v).pdf(x_next)
 
    def sample_observation(self, t, x):
        return x**2/20 + sp.stats.norm(0, self.sigma_w).rvs()
  
    def observation_density(self, t, y, x_sample):
        return sp.stats.norm(x_sample**2/20, self.sigma_w).pdf(y)

