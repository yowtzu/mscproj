import numpy as np
from scipy import stats

class BasicStochasticProcessConfigSingle:
    """ Assume the model of the form
        X_0 ~ N(0, sigma)
        X_k = X_k-1 + W_k-1, p(W) ~ N(0, sigma)
        Y_k = X_k + V_k, p(V) ~ N(0,y_sigma) """
    
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.x_dim = 1
        self.y_dim = 1
        self.mu = np.zeros(self.x_dim)
        self.sigma = np.identity(self.x_dim)
        self.y_mu = np.zeros(self.y_dim)
        self.y_sigma = np.identity(self.y_dim)/10

    def sample_initial(self):
        """ X_0 """
        # return stats.multivariate_normal(self.mu, self.sigma).rvs()
        return np.zeros(1)
    
    def initial_density(self, x):
        """ X_0 """
        return stats.multivariate_normal(self.mu, self.sigma).pdf(x)
 
    def sample_transition(self, x):
        """ X_t | X_t-1 """
        return stats.multivariate_normal(x+self.mu, self.sigma).rvs()

    def transition_density(self, x, x_next):
        """ X_t | X_t-1 """
        return stats.multivariate_normal(self.mu, self.sigma).pdf(x_next-x)

    def sample_observation(self, x):
        """ Y_t | X_t """
        return stats.multivariate_normal(x+self.y_mu, self.y_sigma).rvs()
  
    def observation_density(self, y, x):
        """ Y_t | X_t """
        return stats.multivariate_normal(self.y_mu, self.y_sigma).pdf(y-x)
