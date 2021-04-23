"""
module to provide the Hellekalek benchmark functions
"""

import numpy as np
from scipy import special

from . import WeightFunctions as wf
from . import FacIntegScheme as fis


__all__ = ['get_hellekalek_problem','get_unsym_hellekalek_problem']

def _hellekalek_genunine_integrand(x,alpha):
    return np.power(x,alpha)



# Gauss Legendre case
def _hellekalek_EV_GL(alpha):
    return ((-1)**alpha + 1)/(alpha + 1)/2.0

#hellekalek_GL_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GL,wf.gauss_leg,name='Hellekalek_GL')


# Gauss Chebyshev first order case
def _hellekalek_EV_GC1(alpha):
    return ((-1)**alpha + 1)/np.sqrt(np.pi)/alpha*special.gamma((alpha+1)/2)/special.gamma(alpha/2)

#hellekalek_GC1_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC1,wf.gauss_cheb_1st,name='Hellekalek_GC1')



# Gauss Chebyshev second order case
def _hellekalek_EV_GC2(alpha):
    return ((-1)**alpha + 1)/np.sqrt(np.pi)/2.0*special.gamma((alpha+1)/2)/special.gamma(alpha/2+2)

#hellekalek_GC2_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC2,wf.gauss_cheb_2nd,name='Hellekalek_GC2')


def get_hellekalek_problem(weight):
    if weight == 'GL':
        return fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GL,wf.gauss_leg,name='Hellekalek_GL',para_names=['alpha'])
    if weight == 'GC1':
        return fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC1,wf.gauss_cheb_1st,name='Hellekalek_GC1',para_names=['alpha'])
    if weight == 'GC2':
        return fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC2,wf.gauss_cheb_2nd,name='Hellekalek_GC2',para_names=['alpha'])
    raise NotImplementedError(f"There is no weight function called <{weight}>")




#unsymmetrical

def _unsym_hellekalek_genunine_integrand(x,alpha):
    return _hellekalek_genunine_integrand(x,alpha) + _hellekalek_genunine_integrand(x,alpha-1)



# Gauss Legendre case
def _unsym_hellekalek_EV_GL(alpha):
    return _hellekalek_EV_GL(alpha) + _hellekalek_EV_GL(alpha-1)

#hellekalek_GL_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GL,wf.gauss_leg,name='Hellekalek_GL')


# Gauss Chebyshev first order case
def _unsym_hellekalek_EV_GC1(alpha):
    return _hellekalek_EV_GC1(alpha) + _hellekalek_EV_GC1(alpha-1)

#hellekalek_GC1_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC1,wf.gauss_cheb_1st,name='Hellekalek_GC1')



# Gauss Chebyshev second order case
def _unsym_hellekalek_EV_GC2(alpha):
    return _hellekalek_EV_GC2(alpha) + _hellekalek_EV_GC2(alpha-1)

#hellekalek_GC2_scheme = fis.FactorIntegProblemScheme(_hellekalek_genunine_integrand,_hellekalek_EV_GC2,wf.gauss_cheb_2nd,name='Hellekalek_GC2')


def get_unsym_hellekalek_problem(weight):
    if weight == 'GL':
        return fis.FactorIntegProblemScheme(_unsym_hellekalek_genunine_integrand,_unsym_hellekalek_EV_GL,wf.gauss_leg,name='unsymmetrical Hellekalek_GL',para_names=['alpha'])
    if weight == 'GC1':
        return fis.FactorIntegProblemScheme(_unsym_hellekalek_genunine_integrand,_unsym_hellekalek_EV_GC1,wf.gauss_cheb_1st,name='unsymmetrical Hellekalek_GC1',para_names=['alpha'])
    if weight == 'GC2':
        return fis.FactorIntegProblemScheme(_unsym_hellekalek_genunine_integrand,_unsym_hellekalek_EV_GC2,wf.gauss_cheb_2nd,name='unsymmetrical Hellekalek_GC2',para_names=['alpha'])
    raise NotImplementedError(f"There is no weight function called <{weight}>")
