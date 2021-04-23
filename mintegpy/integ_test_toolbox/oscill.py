"""
OSCILL - Baudin 3.16
"""
import numpy as np

def get_oscill(h,e,seed=123456):
    def oscill(x):
        np.random.seed(seed)
        d = x.shape[-1]
        init_alpha = np.random.uniform(0,1,d)
        init_beta = np.random.uniform(0,1,d)
        omega = d**e/h*np.sum(init_alpha)
        alpha = init_alpha/omega
        beta = init_beta[0]
        exp = 2**d*np.cos(2*np.pi*beta + np.sum(alpha)/2)*np.prod(np.sin(alpha/2)/alpha)
        return np.cos(2*np.pi*beta + np.sum(alpha[None,:]*x,axis=-1))-exp
    return oscill
