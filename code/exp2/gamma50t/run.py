# import logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, linalg
from numpy import zeros, ones, eye, identity, cos, sqrt, exp, sum, prod, arange, pi
from numpy.linalg import inv
import itertools
import timeit
import pickle

n_u = 1

def A(u):
    return eye(n_u)

def C(u):
    return eye(n_u)

def B(u):
    return eye(n_u)

def D(u):
    return eye(n_u)

def F(u): # return a vector
    return u 

def G(u): # return a vector 
    return zeros(n_u)

R = eye(n_u)
Q = eye(n_u)
L = 0.1*eye(n_u)

def gamma(t):
    return 50*(1+t)

# proposal distribution (bootstrapping proposal)
def sample_proposal_u(u_old, gamma):
    """ U_t | U_t-1 """
    return stats.multivariate_normal(zeros(n_u), linalg.inv(L*gamma)).rvs()

def density_proposal_u(u, gamma):
    return exp(-0.5*u.dot(L*gamma).dot(u.T))

def likelihood(y, mu, var, gamma):
    y1 = y-IM
    return exp(-0.5*gamma*y1.dot(linalg.inv(var)).dot(y1.T))

# resample helper function given the index
def resample(data, indices, fromIndex, toIndexExclusive):
    data[fromIndex:toIndexExclusive,:] = data[fromIndex:toIndexExclusive,indices]

def kf_step(X, P, A, Q, F, Y, C, R, G):
    X = np.dot(A, X) + F
    P = np.dot(A, np.dot(P, A.T)) + Q  # Q=BB'
    IM = np.dot(C, X) + G              # G is zero array 
    IS = np.dot(C, np.dot(P, C.T)) + R # R=DD'
    K = np.dot(P, np.dot(C.T, linalg.inv(IS)))
    X = X + np.dot(K, (Y-IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    return (X,P,K,IM,IS)

def cost(IS, IM, y, u, gamma):
    a = likelihood(y, IM, IS, gamma)
    b = density_proposal_u(u, gamma)
    return(a,b)

ess_setting=[0, 1]
mcmc_setting=[0, 1]
TT = [5, 10, 20]
NN = [100, 500, 1000, 5000, 10000]
reps = 30

# for a fixed dimesion, mcmc setting, different time slots (i.e., diff time), different no of particles 
for setting in itertools.product(mcmc_setting, TT, NN, ess_setting):
    #logging.debug("Setting: ", setting)   
    (mcmc_on, T, N, ess_on) = setting

    y= cos(2*pi*0.1*arange(T) + 0.3).reshape(1,T)
    #y= np.array([1,3,4,4,7,8,5,3,5,4,2,1,4,3,5,2,3,3,2,1,1,3,4,4,7,8,5,3,5,4,2,1,4,3,5,2,3,3,2,1]).reshape(1,40)
    y = np.repeat(y, n_u, 0).T
    #y[:,1] = -y[:,1]

    results = dict()    
    for rep in range(reps):

        seed = rep*7 # prime number multiple
        np.random.seed(seed)
        key = setting + (seed,)
        logging.debug("Key : %s", key)
        
        # initialisation
        u = zeros(shape=(T, N, n_u))
        rec_true = zeros((T,N))
        rec2 = zeros((T,N))
        mu = zeros((T,N, n_u))
        Sigma = zeros((T,N, n_u, n_u))       
        accept_ratio = zeros((T,N))
        weight = zeros(N)
        ess = zeros(T)
        
        start_time = timeit.default_timer()
        
        for t in range(T):
            g = gamma(t)
            for i in range(N):
                u[t,i] = sample_proposal_u(u[t-1,i], g)
                # assume mu[-1] and sigma[-1] are zero (i.e., the last ind)
                (mu[t,i],Sigma[t,i],K,IM,IS) = kf_step(mu[t-1,i], Sigma[t-1,i], A(u[t,i]), Q, F(u[t,i]), y[t], C(u[t,i]), R, G(u[t,i]))
                (rec_true[t,i], rec2[t,i]) = cost(IS,IM,y[t,:],u[t,i], g)                     
                weight[i] = rec_true[t,i]
                
            # resampling
            weight = weight/sum(weight)
            ess[t] = sum(weight)**2 / sum(weight**2)
        
            # if ess not on, always samples, if ess on, sample only when ess < threshold
            if (not ess_on) or (ess[t] < (0.5*N)):
                logging.debug("Time Step: %d. ESS=%f" , t, ess[t])
                indices = np.random.choice(range(N),size=N, replace=True, p=weight)
                for data in (u, rec_true, rec2, mu, Sigma):
                    resample(data, indices, 0, t+1)

            # mcmc on
            if mcmc_on:
                logging.debug("MCMC step")
                u_trial = u[0:t+1,0:N,0:n_u] + (0.02*stats.multivariate_normal.rvs(size=(t+1)*N*n_u)).reshape((t+1,N,n_u))
                for i in range(N):                  
                    mu1=zeros((t+1, n_u))
                    Sigma1=zeros((t+1, n_u, n_u))
                    rec_truec = zeros(t+1)
                    rec2c=zeros(t+1)

                    for t in range(t+1): # becareful t is being reused
                        g = gamma(t) # becareful 
                        (mu1[t],Sigma1[t],K,IM,IS) = kf_step(mu1[t-1], Sigma1[t-1], A(u_trial[t,i]), Q, F(u_trial[t,i]), y[t], C(u_trial[t,i]), R, G(u_trial[t,i]))
                        (rec_truec[t], rec2c[t]) = cost(IS,IM,y[t],u_trial[t,i], g) 

                    accept_ratio[t,i] = prod(rec_truec)*prod(rec2c)/(prod(rec_true[0:t+1,i])*prod(rec2[0:t+1,i]))
                    if accept_ratio[t,i] > stats.multivariate_normal.rvs():
                        u[0:t+1,i] = u_trial[:,i]
                        rec_true[0:t+1,i] = rec_truec
                        rec2[0:t+1, i] = rec2c
                        mu[0:t+1,i] = mu1
                        Sigma[0:t+1,i] =Sigma1

        elapsed_time = timeit.default_timer() - start_time
        logging.debug("Elapsed Time: %s", elapsed_time)
        
        # mode
        i = np.nonzero(np.prod(rec_true*rec2, 0) == max(np.prod(rec_true*rec2, 0 )))
        m = u[:,i[0][0], :]
        x2 = np.zeros((T,n_u))
        y2 = np.zeros((T,n_u))
        for t in range(T): 
            x2[t] = A(m[t]).dot(x2[t-1 if (t!=0) else 0])+F(m[t])
            y2[t] = C(m[t]).dot(x2[t])+G(m[t])

        # alternative calculation? normalised by gamma properly
        gamma_m = zeros(shape=(T,N))
        for t in range(T):
            gamma_m[t,:] = gamma(t)

        j_opt = np.sum(np.log(rec_true*rec2)/gamma_m, 0)
        j = np.max(j_opt)
        j_int = np.nonzero(j_opt == j)[0][0]
        m2 = u[:,j_int, :]

        ymean = mu[:,j_int]
        yvar = Sigma[:,j_int]

        results[key] = [m, y2, u, rec_true, rec2, mu, Sigma, ess, accept_ratio, np.mean(np.abs(y2-y)), np.mean((y2-y).T.dot(y2-y)), elapsed_time, m2, j_opt, j, j_int, ymean, yvar, np.mean(np.abs(ymean-y)), np.mean((ymean-y).T.dot(ymean-y))]

    pickle.dump( results, open("res" + str(setting), "wb" ) )
