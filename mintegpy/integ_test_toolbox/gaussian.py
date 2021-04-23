"""
GAUSSIAN - Baudin 3.19
"""
import numpy as np
from scipy.special import erf


def cumGauss(x):
    return .5 + .5*erf(x/np.sqrt(2))

def get_gaussian(alpha,beta):
    def gaussian(x):
        d = x.shape[-1]
        r = cumGauss((1-beta)*np.sqrt(2)*alpha)
        t = cumGauss(-beta*np.sqrt(2)*alpha)
        e = np.prod(np.sqrt(np.pi)/alpha*(r-t))
        return np.exp(-np.sum(alpha[None,:]**2*(x-beta[None,:])**2,axis=-1)) - e
    return gaussian
