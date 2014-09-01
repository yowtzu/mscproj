import matplotlib.pyplot as plt
from numpy import linalg
from collections import Iterable
import numpy as np

class KalmanFilter:
    """ Assume the model of the form
        X_k = AX_k-1 + BU_k + W_k-1, p(W) ~ N(0, R)
        Y_k = HX_k + V_k, p(V) ~ N(0,R) """
    def __init__(self, process):
        self.T = process.T
        self.x_dim = process.config.x_dim
        self.y_dim = process.config.y_dim
        self.xs = process.xs # this is not known, but for plotting purpose
        self.ys = process.ys
        
        # create to store the predictive xs
        self.xs_predictive = np.zeros(shape=(self.T, self.x_dim))
        self.xs_filter = np.zeros(shape=(self.T, self.x_dim))
        
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
        # Model specification
        dim = self.x_dim
        X = np.zeros(dim)    # just random initialsation
        P = np.identity(dim)
        Q = np.identity(dim)
        A = np.identity(dim)
        # A = np.array(((1,0),(1,0)))
        B = np.identity(dim)
        U = np.zeros(dim)
        H = np.identity(dim)
        R = np.identity(dim)
        
        self.xs_predictive[0,:] = X
        self.xs_filter[0,:] = X
        
        # i starts from 1 as that's the first prediction based on x_0
        for i in range(1,self.T): 
             # predict i+1 step value
             (X, P) = self.predict(X, P, A, Q, B, U) 
             self.xs_predictive[i,:] = X  
             # correction
             (X, P, K, IM, IS) = self.update(X, P, self.ys[i, :], H, R)
             self.xs_filter[i,:] = X
             # Y = array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] + abs(0.1 * randn(1)[0])]])
                
    def plot(self):
        """ plot each dimension on different windows/lines """
        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.x_dim)

        if not isinstance(axes, Iterable):
            axes = [axes]
            
        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.xs[:,idx], 'rx', label = 'Hidden Model')
            # ax.plot(range(self.T), self.xs_predictive[:,idx], 'g', label = 'Predictive')
            ax.plot(range(self.T), self.xs_filter[:,idx], 'b', label = 'Filter')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')
            ax.legend(loc='upper left')

        fig.tight_layout()

