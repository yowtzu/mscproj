import scipy as sp
import numpy as np
import scipy.stats

def d_lik(x):
    return sp.stats.norm.pdf(loc=4,scale=2,x=x)

def MH(x0, size, d_lik):
    """Metropolis Hastings algorithm."""

    # proposal settings
    filter_mu = 0
    filter_sigma = 1

    def d_prop(x_old, x_new):
        return sp.stats.norm.pdf(loc=filter_mu, scale=filter_sigma, x=x_old-x_new)

    def r_prop(size, x_old):
        return sp.stats.norm.rvs(loc=x_old + filter_mu, scale=filter_sigma, size=size)

    res = np.zeros(shape = size)
    res[0] = x0

    # generate uniform value in batch
    r_unif = sp.stats.uniform.rvs(size=size-1)

    for i in range(0,size-1):
        res[i+1] = r_prop(1, res[i])
        if r_unif[i] >  min(1,(d_lik(res[i+1])/d_lik(res[i]))*(d_prop(res[i+1], res[i])/d_prop(res[i], res[i+1]))):
            res[i+1] = res[i]

    return res
