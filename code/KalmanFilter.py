import numpy as np
import scipy as sp
from scipy import stats

class StochasticProcess:
    def __init__(self, parameters):
        self.x_dim = parameters.x_dim
        self.y_dim = parameters.y_dim

    def simulate(self, T=100, seed=12345):
        self.xs = np.zeros(T, x_dim)
        self.ys = np.zeros(T, y_dim)
        
        # initialise x[0] and y[0]
        self.xs[0] = self.sample_initial()
        self.ys[0] = self.sample_observation(self.xs[0])
    
        for t in range(1,T):
            self.xs[t] = self.sample_transition(self.xs[t-1])
            self.ys[t] = self.sample_observation(self.xs[t])

    def plot(self, window=None):
        if window:
           pass
 
            
def Process(T=100, sv=1, sw=1, phi=0.95):
    """ process """
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = stats.multivariate_normal.rvs()
    for k in range(1,T):
        x[k] = (phi * x[k-1]) + (sv * stats.multivariate_normal.rvs())
    y = x + (sw*stats.multivariate_normal.rvs(size=T))
    return(x,y)

res = LinearProcess()
y = res[1]

def predict(X, P, A, Q, B, U):
    """ Prediction of X and P
    X: The mean state estimate of the previous step (k-1)
    P: The state covariance of previous step (k-1)
    A: The transition n x n matrix
    Q: The process noise covariance matrix
    B: The input effect matrix
    U: The control input
    """
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return(X,P)

def update(X, P, Y, H, R):
    """ Update correct X state given a new obeservation of Y  
    K: the Kalman Gain matrix
    IM: the Mean of predictive distribution of Y
    IS: The covariance predictive mean of Y
    LH: the predictive probability of measurement
    """
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T)) 
    K = dot(P, dot(H.T, inv(IS))) 
    X = X + dot(K, (Y-IM)) 
    P = P - dot(K, dot(IS, K.T)) 
    LH = gauss_pdf(Y, IM, IS) 
    return (X,P,K,IM,IS,LH) 

def KalmanFilter(T=100, sv=1, sw=1, phi=0.95):
    mp=np.zeros(T)
    mf=np.zeros(T)
    vp=np.zeros(T)
    vf=np.zeros(T)
    my=np.zeros(T)
    vy=np.zeros(T)
    loglike=0
    
    mp[0]=0
    vp[0]=1
    my[0]=mp[0]
    vy[0]=vp[0]+sw**2
    loglike = -0.5*log(2*pi*vy[0])-(0.5*(y[0]-my[0])**2)/vy[0]

    for k in range(0,T-1):
        vf[k]=(sw**2)*vp[k]/(vp[k]+sw**2)
        mf[k]=vf[k]*(mp[k]/vp[k]+y[k]/(sw**2))

        if k < T:
            mp[k+1]=phi*mf[k]
            vp[k+1]=(phi**2)*vf[k]+sv**2
            my[k+1]=mp[k+1]
            vy[k+1]=vp[k+1]+sw**2
            loglike=loglike-0.5*log(2*pi*vy[k+1])-(0.5*(y[k+1]-my[k+1])**2)/vy[k+1]

    return(mp,vp,my,vy)

res2 = KalmanFilter()

# plotting for visualisation
