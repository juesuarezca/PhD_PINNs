"""
module to provide the Gaussian benchmark functions
"""

import numpy as np
from scipy import special

from . import WeightFunctions as wf
from . import FacIntegScheme as fis

__all__ = ['get_gaussian_problem']

def _gaussian_genunine_integrand(x,alpha):
    return np.exp(-np.power(x,2)/np.power(alpha,2))


# Gauss Legendre case

def _gaussian_EV_GL(alpha):
    return alpha*np.sqrt(np.pi)/2*special.erf(1/alpha)

#gaussian_GL_scheme = fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GL,wf.gauss_leg,name='Gaussian_GL')


# Gauss Chebyshev first order case

def _gaussian_EV_GC1(alpha):
    #print("alpha EV",alpha)
    return np.exp(-1/(2*np.power(alpha,2)))*special.i0(1/(2*alpha**2))

#gaussian_GC1_scheme = fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GC1,wf.gauss_cheb_1st,name='Gaussian_GC1')


# Gauss Chebyshev second order case

def _gaussian_EV_GC2(alpha):
    #print("alpha EV",alpha)
    return np.exp(-1/(2*np.power(alpha,2)))*(special.i0(1/(2*alpha**2))+special.i1(1/(2*alpha**2)))

#gaussian_GC2_scheme = fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GC2,wf.gauss_cheb_2nd,name='Gaussian_GC2')


def get_gaussian_problem(weight):
    if weight == 'GL':
        return fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GL,wf.gauss_leg,name='Gaussian_GL',para_names=['alpha'])
    if weight == 'GC1':
        return fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GC1,wf.gauss_cheb_1st,name='Gaussian_GC1',para_names=['alpha'])
    if weight == 'GC2':
        return fis.FactorIntegProblemScheme(_gaussian_genunine_integrand,_gaussian_EV_GC2,wf.gauss_cheb_2nd,name='Gaussian_GC2',para_names=['alpha'])
    raise NotImplementedError(f"There is no weight function called <{weight}>")
