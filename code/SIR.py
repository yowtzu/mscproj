import numpy as np
import scipy as sp

N = 1000 # number of particle
ESS_MIN = 10000000

model_mu = 0
model_sigma = 1
model_nu = 0
model_tau = 1

filter_mu = model_mu
filter_sigma = model_sigma
filter_nu = model_nu
filter_tau = model_tau

def d_init(x):
    return sp.stats.norm(model_mu,model_sigma).pdf(x)

def d_prior(x_old, x_new):
    return sp.stats.norm(model_mu,model_sigma).pdf(x_old-x_new)

def d_lik(x, y):
    return sp.stats.norm(model_nu, model_tau).pdf(y-x)

# proposal distribution
def d_init_prop(x,y):
    return sp.stats.norm(filter_mu, filter_sigma).pdf(x)

def r_init_prop(size, y):
    return sp.stats.norm(filter_mu, filter_sigma).rvs(size)

def d_prop(x_old, x_new, y):
    return sp.stats.norm(filter_mu, filter_sigma).pdf(x_old-x_new)

def r_prop(size, x_old, y):
    return sp.stats.norm(x_old + filter_mu, filter_sigma).rvs(size)

def SIR(data):
    ys = data.ys

    xs = np.zeros(shape = (len(ys), len(ys), N))
    ws = np.zeros(shape = (len(ys), N))
    ess = np.zeros(shape = len(ys))
    
    # step 0 (initialisation)
    xs[0,0,:] = r_init_prop(N, ys[0])
    ws[0,:] = d_init(xs[0,0,:]) * d_lik(xs[0,0,:],ys[0]) / d_init_prop(xs[0,0,:],ys[0])
    ess[0] = sum(ws[0,:])**2 / sum(ws[0,:]**2)
    # not required: ws[t,:] = ws[t,:]/scipy.sum(ws[t,:])
    
    # iteration for t >=1
    for t in range(1,len(ys)):
        print(t)
        print(ess[t-1])
        # resampling if necessary
        if ess[t-1] < (0.5*N):
            print("resampled")
            indices = np.random.choice(range(0,N), N, replace=True, p=ws[t-1,:]/sum(ws[t-1,:]))
            xs[t,:,:] = xs[t-1,:,:][:,indices]
            ws[t,:] = 1
        else:
            xs[t,:,:] = xs[t-1,:,:]
            ws[t,:] = ws[t-1,:]
        
        # next iteration stuff
        xs[t,t,:] = r_prop(N, xs[t,t-1,:], ys[t])
        ws[t,:] = ws[t,:] * d_prior(xs[t,t-1,:], xs[t,t,:]) * d_lik(xs[t,t,:],ys[t]) / d_prop(xs[t,t-1,:],xs[t,t,:],ys[t])

        # normalise the weight and effective sample size
        ws[t,:] = ws[t,:]/sum(ws[t,:])
        ess[t] = sum(ws[t,:])**2 / sum(ws[t,:]**2)
    
    return(xs,ws,ess)
