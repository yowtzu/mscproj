
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import pylab

import matplotlib.pyplot as plt

from scipy import stats
from collections import Iterable


# In[2]:

class BasicStochasticProcessConfig:
    """ Assume the model of the form
        X_0 ~ N(0, sigma)
        X_k = X_k-1 + W_k-1, p(W) ~ N(0, sigma)
        Y_k = X_k + V_k, p(V) ~ N(0,y_sigma) """
    
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.x_dim = 2
        self.y_dim = 2
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

  
class KantasStochasticProcessConfig:
    """ Assume the model of the form in Johassen page 103
        X_0 ~ N(0, 1)
        X_k = X_k-1 + W_k-1, p(W) ~ N(0.9x_k_1 + 0.7, 1)
        Y_k = X_k + V_k, p(V) ~ N(0,0.1)

        X_0 = 0
        X_n = A_n(U_n)X_n-1 + B_n(U_n)W_n + F_n(U_n)
        Y_n = C_n(U_n)X_n + D_n(U_n)V_n + G_n(U_n) """
    
    def __init__(self, seed=10):
        """default parameter is set to be mean = 1.0 2.0, and cov=identity(2)"""
        np.random.seed(seed)
        self.x_dim = 4
        self.y_dim = 4

    def sample_initial(self):
        """ X_0 """
        return 0
    
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


# In[3]:

jconfig = JohansenStochasticProcessConfig(12355)
target = StochasticProcess(jconfig)
#target.plot()

N_SIM = 1000
DIM = 1
T = 150

res3 = np.zeros(shape=[N_SIM, T, DIM])
for i in range(N_SIM):
    conf = JohansenStochasticProcessConfig(54321-3*i)
    sp = StochasticProcess(conf, T=T)
    res3[i] = sp.xs


# In[4]:

sp.mean = np.zeros(shape=[T, DIM])
sp.var = np.zeros(shape=[T, DIM])

for i in range(T):
    sp.mean[i] = np.mean(res3[:,i, :], axis=0)
    sp.var[i] = np.var(res3[:,i, :], axis=0)

# plot mean
plt.plot(sp.var)


# In[5]:

config = BasicStochasticProcessConfig(12355)
target = StochasticProcess(config)
#target.plot()

N_SIM = 1000
DIM = 2
T = 100

resx = np.zeros(shape=[N_SIM, T, DIM])
resy = np.zeros(shape=[N_SIM, T, DIM])
for i in range(N_SIM):
    conf = BasicStochasticProcessConfig(54321-3*i)
    sp = StochasticProcess(conf, T=T)
    resx[i] = sp.xs
    resy[i] = sp.ys


# In[6]:

sp.meanx = np.zeros(shape=[T, DIM])
sp.varx = np.zeros(shape=[T, DIM])
sp.meany = np.zeros(shape=[T, DIM])
sp.vary = np.zeros(shape=[T, DIM])

for i in range(T):
    sp.meanx[i] = np.mean(resx[:,i, :], axis=0)
    sp.varx[i] = np.var(resx[:,i, :], axis=0)
    sp.meany[i] = np.mean(resy[:,i, :], axis=0)
    sp.vary[i] = np.var(resy[:,i, :], axis=0)

plt.plot(sp.varx)


# In[7]:

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
            ax.plot(range(self.T), self.xs[:,idx], 'r')
            ax.plot(range(self.T), self.xs_predictive[:,idx], 'g')
            ax.plot(range(self.T), self.xs_filter[:,idx], 'b')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')

        fig.tight_layout()


# In[8]:

#config = BasicStochasticProcessConfig(135)
jconfig = JohansenStochasticProcessConfig(135)
target = StochasticProcess(jconfig)
#target.plot()

k = KalmanFilter(target)
k.apply()
k.plot()
#np.concatenate((k.xs_filter, k.xs_predictive), axis=1)


# In[9]:

class SIR:
    def __init__(self, process):
        self.T = process.T
        self.x_dim = process.config.x_dim
        self.y_dim = process.config.y_dim
        self.xs = process.xs # this is not known, but for plotting purpose
        self.ys = process.ys
        
        # create to store the predictive xs
        self.xs_predictive = np.zeros(shape=(self.T, self.x_dim))
        self.xs_filter = np.zeros(shape=(self.T, self.x_dim))
        
        # model parameters
        self.N = 1000 # number of particle
        self.ESS_MIN = 10000000
        
        dim = self.x_dim # assume we know the dimension of x
        self.model_mu = np.zeros(dim)
        self.model_sigma = np.identity(dim)
        self.model_nu = np.zeros(dim)
        self.model_tau = np.identity(dim)

        self.filter_mu = np.zeros(dim)
        self.filter_sigma = np.identity(dim)
        self.filter_nu = np.zeros(dim)
        self.filter_tau = np.identity(dim)

    # assume to be known filter stuffs
    ##################
    def d_init(self, x):
        """ X_0 """
        return stats.multivariate_normal(self.model_mu,self.model_sigma).pdf(x)

    def d_prior(self, x_old, x_new):
        """ X_t | X_t-1 """
        return stats.multivariate_normal(self.model_mu,self.model_sigma).pdf(x_new-x_old)

    def d_lik(self, x, y):
        """ Y_t | X_t """
        return stats.multivariate_normal(self.model_nu, self.model_tau).pdf(y-x)

    # proposal distribution (bootstrapping proposal)
    def d_init_prop(self, x,y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).pdf(x)

    def r_init_prop(self, size, y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).rvs(size)

    def d_prop(self, x_old, x_new, y):
        return stats.multivariate_normal(self.filter_mu, self.filter_sigma).pdf(x_new-x_old)

    def r_prop(self, size, x_old, y):
        return stats.multivariate_normal(x_old + self.filter_mu, self.filter_sigma).rvs(size)

    def d_prop(self, x_old, x_new, y):
        return stats.multivariate_normal((self.filter_mu + x_old)/2, self.filter_sigma/2).pdf(x_new)

    def r_prop(self, size, x_old, y):
        return stats.multivariate_normal((x_old + self.filter_mu)/2, self.filter_sigma/2).rvs(size)
   
    
    #################################

    def apply(self):
        # result holder
        self.xshat = np.zeros(shape = (self.T, self.T, self.N))
        self.ws = np.zeros(shape = (self.T, self.N))
        self.ess = np.zeros(shape = self.T)
        
        # self.prediction = np.zeros(shape = len(ys))
        
        # step 0 (initialisation)
        self.xshat[0,0,:] = self.r_init_prop(self.N, self.ys[0])
        self.ws[0,:] = self.d_init(self.xshat[0,0,:]) * self.d_lik(self.xshat[0,0,:],self.ys[0]) / self.d_init_prop(self.xshat[0,0,:],self.ys[0])
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
                self.xshat[t,:,:] = self.xshat[t-1,:,:][:,indices]              
                self.ws[t,:] = 1
            else:
                self.xshat[t,:,:] = self.xshat[t-1,:,:]
                self.ws[t,:] = self.ws[t-1,:]
                
            #print("m xshat", self.xshat[t,:,:])
            #print("m ws", self.ws[t,:])   
            
            # next iteration stuff
            for i in range(self.N):
                self.xshat[t,t,i] = self.r_prop(1, self.xshat[t,t-1,i], self.ys[t])
                self.ws[t,i] = self.ws[t,i] * self.d_prior(self.xshat[t,t-1,i], self.xshat[t,t,i]) * self.d_lik(self.xshat[t,t,i],self.ys[t]) / self.d_prop(self.xshat[t,t-1,i],self.xshat[t,t,i],self.ys[t])

            # self.prediction[t] = np.mean(self.xshat[t,t,:])

            # normalise the weight and effective sample size
            self.ws[t,:] = self.ws[t,:]/sum(self.ws[t,:])
            self.ess[t] = sum(self.ws[t,:])**2 / sum(self.ws[t,:]**2)
            
            #print("e xshat", self.xshat[t,:,:])
            #print("e ws", self.ws[t,:])   
            #print("e Number of effective sample size: ", self.ess[t])
            
    def filter_dist(self, c=0.95):
        # result holder
        xbar = np.zeros(self.T)
        xmax = np.zeros(self.T)
        xmin = np.zeros(self.T)
        
        c_min = (1 - c) / 2
        c_max = 1 - c_min
        print((c_min, c_max))
        
        for t in range(self.T):
            # mean only 
            xbar[t] = sum(self.ws[t,:] * self.xshat[t,t,:]) / sum(self.ws[t,:])
            
            # confidence interval
            sorted_indices = np.argsort(self.xshat[t,t,:])
            sorted_xs = self.xshat[t,t,sorted_indices]
            sorted_cum_ws = np.cumsum(self.ws[t,sorted_indices]/sum(self.ws[t,:]))
            
            min_ind = 0
            max_ind = self.N
            for n in range(self.N):
                # print(self.N, n)
                if sorted_cum_ws[n] < c_min:
                    min_ind = n
                if sorted_cum_ws[self.N - n - 1] > c_max:
                    max_ind = self.N - n - 1
            xmin[t] = sorted_xs[min_ind]
            xmax[t] = sorted_xs[max_ind]
        
        plt.plot(xmin, 'r')
        plt.plot(xbar, 'g')
        plt.plot(xmax, 'r')
        plt.plot(self.xs, 'b')
        
    def smoothing_dist(self, c=0.95):
        # result holder
        xbar = np.zeros(self.T)
        xmax = np.zeros(self.T)
        xmin = np.zeros(self.T)
        
        c_min = (1 - c) / 2
        c_max = 1 - c_min
        print((c_min, c_max))
        
        for t in range(self.T):
            # mean only 
            xbar[t] = sum(self.ws[self.T-1,:] * self.xshat[self.T-1,t,:]) / sum(self.ws[self.T-1,:])
            
            # confidence interval
            sorted_indices = np.argsort(self.xshat[self.T-1,t,:])
            sorted_xs = self.xshat[self.T-1,t,sorted_indices]
            sorted_cum_ws = np.cumsum(self.ws[self.T-1,sorted_indices]/sum(self.ws[self.T-1,:]))
            
            min_ind = 0
            max_ind = self.N
            for n in range(self.N):
                # print(self.N, n)
                if sorted_cum_ws[n] < c_min:
                    min_ind = n
                if sorted_cum_ws[self.N - n - 1 ] > c_max:
                    max_ind = self.N - n - 1
            xmin[t] = sorted_xs[min_ind]
            xmax[t] = sorted_xs[max_ind]
        
        plt.plot(xmin, 'r')
        plt.plot(xbar, 'g')
        plt.plot(xmax, 'r')
        plt.plot(self.xs, 'b')
    
    def predictive_dist(self):
        # result holder
        xbar = np.zeros(self.T)
        xmax = np.zeros(self.T)
        xmin = np.zeros(self.T)
        
        c_min = (1 - c) / 2
        c_max = 1 - c_min
        print((c_min, c_max))
        
        for t in range(self.T):
            # mean only 
            xbar[t] = sum(self.ws[t,:] * self.xshat[t,t,:]) / sum(self.ws[t,:])
            
            # confidence interval
            sorted_indices = np.argsort(self.xshat[t,t,:])
            sorted_xs = self.xshat[t,t,sorted_indices]
            sorted_cum_ws = np.cumsum(self.ws[t,sorted_indices]/sum(self.ws[t,:]))
            
            min_ind = 0
            max_ind = self.N
            for n in range(self.N):
                # print(self.N, n)
                if sorted_cum_ws[n] < c_min:
                    min_ind = n
                if sorted_cum_ws[self.N - n - 1] > c_max:
                    max_ind = self.N - n - 1
            xmin[t] = sorted_xs[min_ind]
            xmax[t] = sorted_xs[max_ind]
        
        plt.plot(xmin, 'r')
        plt.plot(xbar, 'g')
        plt.plot(xmax, 'r')
        plt.plot(self.xs, 'b')
    
    def plot(self):
        """ plot each dimension on different windows/lines """
        y_min = np.min(self.xs)
        y_max = np.max(self.xs)

        fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=self.x_dim)

        if not isinstance(axes, Iterable):
            axes = [axes]
            
        for idx, ax in enumerate(axes):
            ax.plot(range(self.T), self.xs[:,idx], 'r')
            ax.plot(range(self.T), self.xs_predictive[:,idx], 'g')
            ax.plot(range(self.T), self.xs_filter[:,idx], 'y')
            ax.grid(True)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('T')
            ax.set_ylabel('x')
            ax.set_title('hidden states')

        fig.tight_layout()


# In[10]:

config = JohansenStochasticProcessConfig(1345)
target = StochasticProcess(config)

sir = SIR(target)
sir.apply()

sir.filter_dist(c=0.95)



# In[11]:

sir.smoothing_dist(c=0.95)

#print(sir.xshat[30,30,:].shape)
#plt.hist(sir.xshat[10,10,:])


# In[ ]:



