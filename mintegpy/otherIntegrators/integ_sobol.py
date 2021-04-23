"""
sobol squence Integrator
"""
__all__ = ['integrate_sobol']
import numpy as np
import sobol_seq
import mintegpy as mt
from mintegpy.diagnostics import count_class


class sobol(object):
    def __init__(self,dim, num_pts):
        """
        only use at [0,1]^dim
        """
        self.__pts = sobol_seq.i4_sobol_generate(dim,num_pts)
        self.__n = num_pts

    @property
    def pts(self):
        return self.__pts

    @property
    def num_pts(self):
        return self.__n

    def integrate(self,f):
        return np.sum(f(self.pts))/self.num_pts

def integrate_sobol(func_getter,func_args = None,dim = 2,neval=100):
    if func_args is None:
        temp_func = func_getter()
    else:
        temp_func = func_getter(**func_args)

    temp_sobol = sobol(dim,neval)

    @count_class()
    def temp_integrand(x):
        return temp_func(x)

    res = temp_sobol.integrate(temp_integrand)

    return {'res':max(abs(res),mt.MACHINE_PRECISION),'count':temp_integrand.called}
