import numpy as np
import pylab
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt

from scipy import stats
from collections import Iterable

class StochasticProcess:
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.mu = np.array([0.0, 0.0])
        self.sigma = np.identity(2)
        self.y_mu = np.array([0.0, 0.0])
        self.y_sigma = np.identity(2)
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

class KalmanFilter:    
    def __init__(self, sp):
        self.T = sp.T
        self.x_dim = sp.x_dim
        self.y_dim = sp.y_dim
        self.xs = sp.xs
        self.ys = sp.ys
        
        self.xshat = np.zeros(shape=(self.T, self.x_dim))
        
    def predict(self, X, P, A, Q, B, U):
        """ Prediction of X and P                                                                     
        X: The mean state estimate of the previous step (k-1)                                         
        P: The state covariance of previous step (k-1)                                                
        A: The transition n x n matrix                                                                
        Q: The process noise covariance matrix                                                        
        B: The input effect matrix                                                                    
        U: The control input                                                                          
        """
        X = np.dot(A, X) + np.dot(B, U)
        P = np.dot(A, np.dot(P, A.T)) + Q
        return(X,P)

    def update(self, X, P, Y, H, R):
        """ Update correct X state given a new obeservation of Y                                      
        K: the Kalman Gain matrix                                                                     
        IM: the Mean of predictive distribution of Y                                                  
        IS: The covariance predictive mean of Y                                                       
        LH: the predictive probability of measurement                                                 
        """
        IM = np.dot(H, X)
        IS = R + np.dot(H, np.dot(P, H.T))
        K = np.dot(P, np.dot(H.T, linalg.inv(IS)))
        X = X + np.dot(K, (Y-IM))
        P = P - np.dot(K, np.dot(IS, K.T))
        return (X,P,K,IM,IS)
    
    def apply(self):
        # Applying the Kalman Filter
        X = np.zeros(2)
        P = np.identity(2)
        Q = np.identity(2)
        A = np.identity(2)
        B = np.identity(2)
        U = np.zeros(2)
        H = np.zeros(2)
        R = np.identity(2)
        for i in range(0,self.T): 
             (XMinus, PMinus) = self.predict(X, P, A, Q, B, U) # predict i+1 step value 
             self.xshat[i] = XMinus
             (X, P, K, IM, IS) = self.update(XMinus, PMinus, self.ys[i], H, R)
             # Y = array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] + abs(0.1 * randn(1)[0])]])
    
x = StochasticProcess(34)
x.simulate()
x.plot()
print(x.T)
k = KalmanFilter(x)
k.xshat
k.apply()
