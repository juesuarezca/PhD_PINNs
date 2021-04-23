"""
benchmark function for mintegpy
"""
__all__ = ['integrate_mintegpy']
import numpy as np
import mintegpy as mt
import mintegpy.minterpy.utils as utils
from mintegpy.diagnostics import count_class



def integrate_Hellekalek_our(dim,alpha,polydeg,lp = 1):
    temp_integrator = integrate(dim,polydeg,lp,0,1)
    temp_hellekalek = get_Hellekalek_nD(alpha)
    @count_class(axis=-1)
    def hellekalek_our(x):
        return temp_hellekalek(x.T)

    result = temp_integrator.integrate(hellekalek_our)
    if result<eps:
        result = eps
    return result, hellekalek_our.called



def integrate_mintegpy(func_getter,func_args = None,dim=2,polydeg=2,lp=1,point_kind='leja',optimal = False,use_data = False,split = None,weight_function = 'legendre'):
    if func_args is None:
        temp_func = func_getter()
    else:
        temp_func = func_getter(**func_args)

    @count_class(axis=-1)
    def temp_integrand(x):
        return temp_func(x.T)

    #print("split integrate_mintegpy",split)
    temp_integrator = mt.integrate(dim,polydeg,lp,0,1,use_data=use_data,point_kind=point_kind,optimal=optimal,split = split,weight_function = weight_function)
    result = temp_integrator.integrate(temp_integrand)

    return {'res':max(abs(result),mt.MACHINE_PRECISION),'count':temp_integrand.called,'integrator':temp_integrator}
