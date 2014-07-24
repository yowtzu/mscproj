import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Iterable
from numpy import linalg

class KantasSIR:
    def __init__(self, process, N=30, resample_move_enabled=False):
        self.T = process.T
        self.x_dim = process.config.x_dim
        self.y_dim = process.config.y_dim
        self.xs = process.xs # this is not known, but for plotting purpose only
        self.ys = process.ys
        
        # Model specification
        dim = self.x_dim
        self.X = np.zeros(dim)    # just random initialsation
        self.P = np.identity(dim)
        self.F = np.identity(dim)
        self.G = np.identity(dim)
        self.C = np.identity(dim)
        self.A = np.identity(dim)
        self.B = np.identity(dim)
        self.Y = np.identity(dim)
        self.U = np.zeros(dim)
        self.H = np.identity(dim)
        self.D = np.identity(dim)

        # model parameters
        self.N = N  # number of particle
        
        # assume we know the dimension of x
        self.model_mu = np.zeros(self.x_dim)
        self.model_sigma = np.identity(self.x_dim)
        self.model_nu = np.zeros(self.y_dim)
        self.model_tau = np.identity(self.y_dim)

        self.filter_mu = np.zeros(self.x_dim)
        self.filter_sigma = np.identity(self.x_dim)
        self.filter_nu = np.zeros(self.y_dim)
        self.filter_tau = np.identity(self.y_dim)
        
        self.resample_move_enabled = resample_move_enabled

    # assume to be known filter stuffs
    ##################
    def d_init(self, x):
        """ X_0 """
        return stats.multivariate_normal(self.model_mu,self.model_sigma).pdf(x)

    def d_prior(self, x_old, x_new):
        """ X_t | X_t-1 """
        return stats.multivariate_normal(self.model_mu,self.model_sigma).pdf(x_new-x_old)

    def d_lik(self, x, y, t):
        """ Y_t | X_t """
        """ need to perform Kalman filter """
        # use the predictive X and P
        return stats.multivariate_normal(self.IM, self.IS).pdf(y)

    # proposal distribution (bootstrapping proposal)
    def d_init_prop(self, x,y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).pdf(x)

    def r_init_prop(self, size, y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).rvs(size).reshape(self.N, self.x_dim)


    def d_prop(self, x_old, x_new, y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).pdf(x_new-x_old)

    def r_prop(self, size, x_old, y):
        r = stats.multivariate_normal(x_old + self.filter_mu, self.filter_sigma).rvs(size)
        return(r)

    def predict(self, X, P, A, B, F, U):
        """ Prediction of X and P                                                                     
        X: The mean state estimate of the previous step (k-1)                                         
        P: The state covariance of previous step (k-1)                                                
        A: The transition n x n matrix                                                                
        Q: The process noise covariance matrix                                                        
        B: The input effect matrix                                                                    
        U: The control input                                                                          
        """
        X = np.dot(A, X) + np.dot(F, U)
        P = np.dot(A, np.dot(P, A.T)) + np.dot(B, B.T)
        return(X,P)

    def update(self, X, P, Y, C, G, U, D):
        """ Update correct X state given a new obeservation of Y                                      
        K: the Kalman Gain matrix                                                                     
        IM: the Mean of predictive distribution of Y                                                  
        IS: The covariance predictive mean of Y                                                       
        LH: the predictive probability of measurement                                                 
        """
        IM = np.dot(C, X) + np.dot(G, U)
        IS = np.dot(C, np.dot(P, C.T)) + np.dot(D, D.T)
        K = np.dot(P, np.dot(C.T, linalg.inv(IS)))
        X = X + np.dot(K, (Y-IM))
        P = P - np.dot(K, np.dot(C, P))
        return(X,P,K,IM,IS)
    
    #################################
    def apply(self):
        # result holder
        self.xshat = np.zeros(shape = (self.T, self.T, self.N, self.x_dim)) # changed this line
        self.ws = np.zeros(shape = (self.T, self.N))
        self.ess = np.zeros(shape = self.T)
        
        # step 0 (initialisation)
        self.xshat[0,0] = self.r_init_prop(self.N, self.ys[0])

        # MODIFY!!! 
        self.X = np.dot(self.A, self.X) + np.dot(self.F, self.U)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + np.dot(self.B, self.B.T)
        self.IM = np.dot(self.C, self.X) + np.dot(self.G, self.U)
        self.IS = np.dot(self.C, np.dot(self.P, self.C.T)) + np.dot(self.D, self.D.T)
        
        self.ws[0,:] = self.d_init(self.xshat[0,0,:]) * self.d_lik(self.xshat[0,0,:],self.ys[0], 0) / self.d_init_prop(self.xshat[0,0,:],self.ys[0])

        self.K = np.dot(self.P, np.dot(self.C.T, linalg.inv(self.IS)))
        self.X = self.X + np.dot(self.K, (self.Y-self.IM))
        self.P = self.P - np.dot(self.K, np.dot(self.C, self.P))


        self.ess[0] = sum(self.ws[0,:])**2 / sum(self.ws[0,:]**2)

        # not required: ws[t,:] = ws[t,:]/scipy.sum(ws[t,:])
        
        #print("Iteration: ", 0)
        #print("xshat", 0, self.xshat[0,:,:])
        #print("ws", 0, self.ws[0,:])  
        #print("Number of effective sample size: ", self.ess[0])
        
        # iteration for t >=1
        for t in range(1,self.T):
            #print("Iteration: ", t)

            # resampling if necessary
            if self.ess[t-1] < (0.5 * self.N):
                # print("re-sample")
                indices = np.random.choice(range(self.N),size=self.N, replace=True, p=self.ws[t-1,:]/sum(self.ws[t-1,:]))
                self.xshat[t,:,:,:] = self.xshat[t-1,:,:,:][:,indices]              
                self.ws[t,:] = 1
            else:
                self.xshat[t,:,:,:] = self.xshat[t-1,:,:,:]
                self.ws[t,:] = self.ws[t-1,:]
                
            #print("m xshat", self.xshat[t,:,:])
            #print("m ws", self.ws[t,:])   
            
            self.X = np.dot(self.A, self.X) + np.dot(self.F, self.U)
            self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + np.dot(self.B, self.B.T)
            self.IM = np.dot(self.C, self.X) + np.dot(self.G, self.U)
            self.IS = np.dot(self.C, np.dot(self.P, self.C.T)) + np.dot(self.D, self.D.T)
            
            # next iteration stuff
            for i in range(self.N):
                self.xshat[t,t,i,:] = self.r_prop(1, self.xshat[t,t-1,i,:], self.ys[t,:])

                # MODIFY!!!
                self.ws[t,i] = self.ws[t,i] * self.d_prior(self.xshat[t,t-1,i,:], self.xshat[t,t,i,:]) * self.d_lik(self.xshat[t,t,i,:],self.ys[t,:], t) / self.d_prop(self.xshat[t,t-1,i,:],self.xshat[t,t,i,:],self.ys[t,:])

            self.K = np.dot(self.P, np.dot(self.C.T, linalg.inv(self.IS)))
            self.X = self.X + np.dot(self.K, (self.ys[t,:]-self.IM))
            self.P = self.P - np.dot(self.K, np.dot(self.C, self.P))

            # normalise the weight and effective sample size
            self.ws[t,:] = self.ws[t,:]/sum(self.ws[t,:])
            self.ess[t] = sum(self.ws[t,:])**2 / sum(self.ws[t,:]**2)
            
            #print("e xshat", self.xshat[t,:,:])
            #print("e ws", self.ws[t,:])   
            #print("e Number of effective sample size: ", self.ess[t])

            # resample move
            if self.resample_move_enabled:
                pass

    def MH(x0, size=11):
        """Metropolis Hastings algorithm.
        x0 is the initial point that has density >=0 on the target distribution
        x0 is one dimensional vector
        d_lik is the target distribution
        size is the sample set one would like to generate (i.e., we only move size-1 step, as x0 is the first sample
        """

        # proposal settings
        proposal_mu = np.zeros(len(x0))
        proposal_sigma = np.identity(len(x0))
    
        def d_prop(x_old, x_new):
            """xx is a one dimensional vector (for generalisation sake)"""
            #return stats.norm.pdf(loc=filter_mu, scale=filter_sigma, x=x_old-x_new)
            return stats.multivariate_normal.pdf(x=x_new-x_old, mean=proposal_mu, cov=proposal_sigma)
            
        def r_prop(x_old):
            #return stats.norm.rvs(loc=x_old + filter_mu, scale=filter_sigma, size=size)
            return stats.multivariate_normal.rvs(mean=x_old+proposal_mu, cov=proposal_sigma)
            
        # initialisation
        res = np.zeros(shape = [size, len(x0)]) # first dim:time, second dim: x
        res[0] = current = x0
 
        # the first value is not used at all
        u = stats.uniform.rvs(size = size)
        
        # iterate through
        for i in range(1,size):
            prop = r_prop(current)
            if u[i] < min(1,(d_lik(prop)/d_lik(current)) *(d_prop(prop, current)/d_prop(current, prop))):
                current = prop
            res[i] = current
                
        return res[:,-1]
            
    def filter_dist(self, c=0.95):
        # result holder
        xbar = np.zeros(shape=(self.T, self.x_dim))
        #xmax = np.zeros(shape=(self.T, self.x_dim))
        #xmin = np.zeros(shape=(self.T, self.x_dim))
        
        #c_min = (1 - c) / 2
        #c_max = 1 - c_min
        #print((c_min, c_max))
        
        for t in range(self.T):
            # mean only 
            xbar[t,:] = np.dot(self.ws[t,:], self.xshat[t,t,:,:]) / sum(self.ws[t,:])
            
            # confidence interval
            #sorted_indices = np.argsort(self.xshat[t,t,:,:], axis = 0)
            #sorted_xs = sir.xshat[t,t,:,:][sorted_indices, range(self.x_dim)]
            

            #sorted_cum_ws = np.cumsum(self.ws[t,:][sorted_indices, range(self.x_dim)]/sum(self.ws[t,:]))
            
            #min_ind = 0
            #max_ind = self.N
            #for n in range(self.N):
            #    # print(self.N, n)
            #    if sorted_cum_ws[n] < c_min:
            #        min_ind = n
            #    if sorted_cum_ws[self.N - n - 1] > c_max:
            #        max_ind = self.N - n - 1
            #xmin[t] = sorted_xs[min_ind]
            #xmax[t] = sorted_xs[max_ind]

        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.x_dim)

        if not isinstance(axes, Iterable):
            axes = [axes]

        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.xs[:,idx], 'rx', label = 'Hidden Model')
            #ax.plot(range(self.T), xmin, 'g', label = 'lower')
            ax.plot(range(self.T), xbar[:,idx], 'b', label = 'smoothing')
            #ax.plot(range(self.T), xmax, 'g', label = 'upper')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')
            ax.legend(loc='upper left')
        
        fig.tight_layout()
        
    def smoothing_dist(self, c=0.95):
        # result holder
        xbar = np.zeros(shape=(self.T, self.x_dim))

        for t in range(self.T):
            # mean only 
            xbar[t,:] = np.dot(self.ws[self.T-1,:], self.xshat[self.T-1,t,:,:]) / sum(self.ws[self.T-1,:])
            #xbar[t] = sum(self.ws[self.T-1,:] * self.xshat[self.T-1,t,:]) / sum(self.ws[self.T-1,:])
            
            # confidence interval
            #sorted_indices = np.argsort(self.xshat[self.T-1,t,:])
            #sorted_xs = self.xshat[self.T-1,t,sorted_indices]
            #sorted_cum_ws = np.cumsum(self.ws[self.T-1,sorted_indices]/sum(self.ws[self.T-1,:]))
            
            #min_ind = 0
            #max_ind = self.N
            #for n in range(self.N):
            #    # print(self.N, n)
            #    if sorted_cum_ws[n] < c_min:
            #        min_ind = n
            #    if sorted_cum_ws[self.N - n - 1 ] > c_max:
            #        max_ind = self.N - n - 1
            #xmin[t] = sorted_xs[min_ind]
            #xmax[t] = sorted_xs[max_ind]

        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.x_dim)

        if not isinstance(axes, Iterable):
            axes = [axes]

        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.xs[:,idx], 'rx', label = 'Hidden Model')
            #ax.plot(range(self.T), xmin, 'g', label = 'lower')
            ax.plot(range(self.T), xbar[:,idx], 'b', label = 'smoothing')
            #ax.plot(range(self.T), xmax, 'g', label = 'upper')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')
            ax.legend(loc='upper left')
        
        fig.tight_layout()

