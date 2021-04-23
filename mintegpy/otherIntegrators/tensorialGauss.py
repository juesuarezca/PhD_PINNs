"""
module for tensorial gaussian cubature
"""
import numpy as np
from scipy import special
from mintegpy.helper import cartesian_product

__all__ = ['TensorialGauss']


def transform_points(arr,dom):
    slope = (dom[:,1] - dom[:,0])/2
    offset = (dom[:,1] + dom[:,0])/2
    return slope[None,:]*arr + offset[None,:]

def transfrom_weights(arr,dom):
    slope = (dom[:,1] - dom[:,0])/2
    return slope[None,:]*arr

class TensorialGauss(object):
    __avial_weights = ['leg','cheb1st','cheb2nd']
    __avial_1Dgen = {
        'leg':np.polynomial.legendre.leggauss,
        'cheb1st':np.polynomial.chebyshev.chebgauss,
        'cheb2nd':special.roots_chebyu
    }
    def __init__(self,deg1D,dim,weight_fct = None,domain = None):
        if domain is None:
            self.__domain = np.tile([-1,1],(dim,1))
        else:
            self.__domain = np.atleast_2d(domain)
        if weight_fct is None:
            self.__weight_fct = 'leg'
        elif weight_fct in self.__avial_weights:
            self.__weight_fct = weight_fct
        else:
            raise ValueError(f"There is no weight_function called <{weight_fct}>")

        self.__deg1D = deg1D
        self.__dim = dim
        if deg1D**dim>1e6:
            print(f"Warning: large problem size detected ({deg1D**dim} points). Memory and time consuming calculations.")
        self.__build_points_weights()

    def __build_points_weights(self):
        points1D, weights1D = self.__avial_1Dgen[self.__weight_fct](self.__deg1D)
        self.__points = cartesian_product(*[points1D]*self.__dim)
        self.__weights = cartesian_product(*[weights1D]*self.__dim)


    @property
    def points(self):
        return transform_points(self.__points,self.__domain)

    @property
    def size(self):
        #returns number of points
        return self.__points.shape[0]

    @property
    def weights(self):
        return np.prod(transfrom_weights(self.__weights,self.__domain),axis = -1)

    def integrate(self,func):
        return np.dot(self.weights,func(self.points))
