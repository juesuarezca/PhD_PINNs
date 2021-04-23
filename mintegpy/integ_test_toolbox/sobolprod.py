"""
SOBOLPROD - Baudin 3.15
"""
import numpy as np

def get_sobolprod():
    def sobolprod(x):
        d = x.shape[-1]
        idx = np.arange(d)+1
        mu = np.ones(d)
        e = np.prod(mu)
        gamma_square = 1/(3*(idx +1)**2)
        nu = np.prod((mu**2 + gamma_square)) - np.prod(mu**2)
        facs = (idx[None,:] + 2*x)/(idx[None,:]+1)
        return np.array((np.prod(facs,axis=-1) - e)/(np.sqrt(nu)))
    return sobolprod

if __name__=='__main__':
    test_func = get_sobolprod()
    test_x = np.linspace(0,1,100).reshape(5,20)
    test_vals = test_func(test_x)
    print(test_vals.shape)
    print(test_vals)
