"""
integration using scipy quad
"""
import numpy as np
from scipy.integrate import nquad
import mintegpy as mt

def integrate_quad(func_getter,func_args = None,dim = 2,quad_limit = 50):
    if func_args is None:
        temp_func = func_getter()
    else:
        temp_func = func_getter(**func_args)

    def integrand(*args):
        temp_x = np.atleast_2d(np.array(args))
        return temp_func(temp_x)
    res_quad,err_quad,opt = nquad(integrand,[[0,1]]*dim,full_output=True,opts = {'epsabs': 1.49e-12,'epsrel': 1.49e-12,'limit':quad_limit})#,'maxp1':10})
    return {'res':max(abs(res_quad),mt.MACHINE_PRECISION),'err':err_quad,'count':opt['neval']}
