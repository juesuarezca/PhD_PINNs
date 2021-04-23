"""
module to provide several weight functions
"""
import numpy as np


class WeightFactory(object):
    def __init__(self,weight_func,name = None):
        if name is None:
            self.name = 'default'
        else:
            self.name = name

        self.func = weight_func

    def __call__(self,x):
        return self.func(x)

    def __repr__(self):
        return self.name

def __gen_gauss_cheb_1st(x):
    x = np.require(x)
    if len(x.shape)==1:
        x = np.atleast_2d(x).reshape(-1,1)
    return np.prod(1/np.sqrt(1-x**2),axis = -1)

gauss_cheb_1st = WeightFactory(__gen_gauss_cheb_1st,'gauss_cheb_1st')


def __gen_gauss_cheb_2nd(x):
    x = np.require(x)
    if len(x.shape)==1:
        x = np.atleast_2d(x).reshape(-1,1)
    return np.prod(np.sqrt(1-x**2),axis = -1)

gauss_cheb_2nd = WeightFactory(__gen_gauss_cheb_2nd,'gauss_cheb_2nd')

def __gen_gauss_leg(x):
    return 1.0

gauss_leg = WeightFactory(__gen_gauss_leg,'gauss_leg')
