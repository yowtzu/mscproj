from scipy import stats
import numpy as np

def d_lik(x):
    """the target likelihood function we try to sample from
    x is a one dimensional vector (for generalisation sake)"""
    # stats.norm.pdf(loc=4,scale=2,x=x)
    mean = np.zeros(len(x))
    cov = np.identity(len(x))
    return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)

def MH(x0, size, d_lik):
    """Metropolis Hastings algorithm.
       x0 is the initial point that has density >=0 on the target distribution
       x0 is one dimensional vector
       size is the sample set one would like to generate (i.e., we only move size-1 step, as x0 is the first sample
    """

    # proposal settings
    proposal_mu = (len(x0))
    proposal_sigma = np.identity(len(x0))

    def d_prop(x_old, x_new):
        """xx is a one dimensional vector (for generalisation sake)"""
        #return stats.norm.pdf(loc=filter_mu, scale=filter_sigma, x=x_old-x_new)
        return stats.multivariate_normal.pdf(x=x_new-x_old, mean=proposal_mu, cov=proposal_sigma)

    def r_prop(x_old):
        #return stats.norm.rvs(loc=x_old + filter_mu, scale=filter_sigma, size=size)
        return stats.multivariate_normal.rvs(mean=x_old+proposal_mu, cov=proposal_sigma)

    # initialisation
    res = np.zeros(shape = [size, len(x)]) # first dim:time, second dim: x 
    res[0,:] = x0

    # generate uniform value in batch
    r_unif = stats.uniform.rvs(size=[size-1, len(x)]) 

    # iterate through
    for i in range(0,size-1):
        res[i+1] = r_prop(res[i])
        if r_unif[i] >  min(1,(d_lik(res[i+1])/d_lik(res[i]))*(d_prop(res[i+1], res[i])/d_prop(res[i], res[i+1]))):
            res[i+1] = res[i]

    return res
