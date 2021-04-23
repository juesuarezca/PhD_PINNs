"""
vegas monte carlo integration (vegas package from Lepage)
"""
__all__ = ['integrate_vegas']
import numpy as np
import vegas
import mintegpy as mt
import mintegpy.minterpy.utils as utils
from mintegpy.diagnostics import count_class
import gvar as gv


def integrate_vegas(func_getter,func_args = None,dim = 2,iterations=10,evaluations = 1000,do_trainig = True):
    if func_args is None:
        temp_func = func_getter()
    else:
        temp_func = func_getter(**func_args)

    @count_class()
    @vegas.batchintegrand
    def integrand_batch(x):
        return temp_func(np.atleast_2d(x))

    integ = vegas.Integrator(dim*[[0,1]])
    calc_evals = evaluations

    if do_trainig:
        calc_evals = np.floor(evaluations/2)
        train_evals = evaluations - calc_evals
        training = integ(integrand_batch,nitn = iterations,neval = train_evals)
    else:
        training = None
    result = integ(integrand_batch,nitn = iterations,neval = evaluations)
    del(integ)
    #print(abs(gv.mean(result)))
    return {'res':max(abs(gv.mean(result)),mt.MACHINE_PRECISION),'sdev':gv.sdev(result),'training':training,'count':integrand_batch.called}
