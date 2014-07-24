import numpy as np
from scipy import stats

class JohansenStochasticProcessConfig:
    """ Assume the model of the form in Johassen page 103
        X_0 ~ N(0, 1)
        X_k = X_k-1 + W_k-1, p(W) ~ N(0.9x_k_1 + 0.7, 1)
        Y_k = X_k + V_k, p(V) ~ N(0,0.1) """
        
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.x_dim = 1
        self.y_dim = 1

    def sample_initial(self):
        """ X_0 """
        return stats.multivariate_normal(0, 1).rvs()

    def initial_density(self, x):
        """ X_0 """
        return stats.multivariate_normal(0, 1).pdf(x)
 
    def sample_transition(self, x):
        """ X_t | X_t-1 """
        return stats.multivariate_normal((0.9*x)+0.7, 1).rvs()

    def transition_density(self, x, x_next):
        """ X_t | X_t-1 """
        return stats.multivariate_normal((0.9*x)+0.7, 1).pdf(x_next)

    def sample_observation(self, x):
        """ Y_t | X_t """
        r = stats.multivariate_normal(x, 0.1).rvs()
        return r
  
    def observation_density(self, y, x):
        """ Y_t | X_t """
        r = stats.multivariate_normal(0, 0.1).pdf(y-x)
        return r 
