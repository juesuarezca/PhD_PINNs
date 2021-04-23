"""
module to provide the Oscil benchmark functions
"""

import numpy as np
from scipy import special

from . import WeightFunctions as wf
from . import FacIntegScheme as fis


__all__ = ['get_oscil_problem']

def _oscil_genunine_integrand(x,alpha,beta):
    return np.sin(np.pi*alpha*x + beta)


# Gauss Legendre case

def _oscil_EV_GL(alpha,beta):
    return 1/alpha/np.pi*np.sin(np.pi*alpha)*np.sin(beta)

#oscil_GL_scheme = fis.FactorIntegProblemScheme(_oscil_genunine_integrand,_oscil_EV_GL,wf.gauss_leg,name='Oscil_GL')


# Gauss Chebyshev first order case

def _oscil_EV_GC1(alpha,beta):
    return special.j0(np.pi*alpha)*np.sin(beta)

#oscil_GC1_scheme = fis.FactorIntegProblemScheme(_oscil_genunine_integrand,_oscil_EV_GC1,wf.gauss_cheb_1st,name='Oscil_GC1')

# Gauss Chebyshev second order case

def _oscil_EV_GC2(alpha,beta):
    return 2/np.pi/alpha*special.j1(np.pi*alpha)*np.sin(beta)

#oscil_GC2_scheme = fis.FactorIntegProblemScheme(_oscil_genunine_integrand,_oscil_EV_GC2,wf.gauss_cheb_2nd,name='Oscil_GC2')

def get_oscil_problem(weight):
    if weight == 'GL':
        return fis.FactorIntegProblemScheme(_oscil_genunine_integrand,_oscil_EV_GL,wf.gauss_leg,name='Oscil_GL',para_names=['alpha','beta'])
    if weight == 'GC1':
        return fis.FactorIntegProblemScheme(_oscil_genunine_integrand,_oscil_EV_GC1,wf.gauss_cheb_1st,name='Oscil_GC1',para_names=['alpha','beta'])
    if weight == 'GC2':
        return fis.FactorIntegProblemScheme(_oscil_genunine_integrand, _oscil_EV_GC2, wf.gauss_cheb_2nd,name='Oscil_GC2',para_names=['alpha','beta'])
    raise NotImplementedError(f"There is no weight function called <{weight}>")
