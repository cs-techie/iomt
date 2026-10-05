import numpy as np

def gaussian_sigma(sensitivity,epsilon,delta): return sensitivity*np.sqrt(2*np.log(1.25/delta))/epsilon

def add_gaussian_dp(values,sensitivity=1,epsilon=1,delta=1e-5,rng=None):
    rng=rng or np.random.default_rng(); sigma=gaussian_sigma(sensitivity,epsilon,delta); a=np.asarray(values,dtype=float)
    return a+rng.normal(0,sigma,size=a.shape),sigma
