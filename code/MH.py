from scipy import stats
import numpy as np
 
def d_lik(x):
    """the target likelihood function we try to sample from
    x is a one dimensional vector (for generalisation sake)"""
    # stats.norm.pdf(loc=4,scale=2,x=x)
    mean = np.zeros(len(x))
    #mean = np.array([0.0, 1.0])
    mean = [-1.0, -0.2]
    cov = np.identity(len(x))
    cov = np.array([[1.0, 0.3], [0.3, 1]])
    return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)
 
def MH(x0, size, d_lik):
    """Metropolis Hastings algorithm.
       x0 is the initial point that has density >=0 on the target distribution
       x0 is one dimensional vector
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
    for i in range(1,size-1):
        prop = r_prop(current)
        if u[i] < min(1,(d_lik(prop)/d_lik(current)) *(d_prop(res[i+1], res[i])/d_prop(res[i], res[i+1]))):
            current = prop
        res[i] = current
 
    return res
 
res = MH([-0.4, -0.43], 50000, d_lik)
 
print("The mean of the x0 is %f" % np.mean(res[:,0]))
print("The mean of the x1 is %f" % np.mean(res[:,1]))
np.cov(np.transpose(res))
