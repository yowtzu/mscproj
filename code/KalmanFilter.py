import numpy as np
from scipy import stats

def LinearProcess(T=100, sv=1, sw=1, phi=0.95):
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = stats.multivariate_normal.rvs()
    for k in range(1,T):
        x[k] = (phi * x[k-1]) + (sv * stats.multivariate_normal.rvs())
    y = x + (sw*stats.multivariate_normal.rvs(size=T))
    return(x,y)

res = LinearProcess()
y = res[1]

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
