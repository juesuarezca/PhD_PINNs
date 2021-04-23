"""
PRODEXP - Baudin 3.5
"""
import numpy as np

def get_prodexp():
    w = np.sqrt((15*np.exp(15) + 15)/(13*np.exp(15) + 17))
    def prodexp(x):
        d = x.shape[-1]
        facs = (np.exp(30*x-15)-1)/(np.exp(30*x-15)+1)
        return w**d*np.prod(facs,axis=-1)
    return prodexp
