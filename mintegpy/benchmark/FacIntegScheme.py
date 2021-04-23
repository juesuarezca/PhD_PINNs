"""
Module for the Factorised integration problem scheme
"""

import numpy as np
from . import WeightFunctions as wf
from mintegpy.diagnostics import count_class

class FactorIntegProblemScheme(object):
    qualified_names = {
        'name':'name',
        'paras':'parameters',
        'weight':'weight function',
        'gen_integ':'genuine inegrand',
        'ev':'expectation value term',
        'dim':'current dimension'
    }

    def __init__(self,genuine_integrand, ev_term, weight, name = None, para_names = None):
        self.__gen_integrand = genuine_integrand
        self.__ev = ev_term
        self.__weight = weight
        self.__name = name
        self.para_names = para_names
        self.dim = None
        self.neval = 0


    def ev_term(self,alpha):
        return self.__ev(alpha)

    def genuine_integrand(self,x,alpha):
        return self.__gen_integrand(x,alpha)

    @property
    def info(self):
        info_dict = {
            'name': f"{self.name}",
            'paras': f": {self.para_names}",
            'weight': f"{self.weight_name}",
            'gen_integ': f"{str(self.__gen_integrand)}",
            'ev': f"{str(self.__ev)}",
            'dim': f"{self.dim}"
        }

        return info_dict

    def print_info(self):
        print("\n".join([f"{self.qualified_names[key]} : {item}" for key,item in self.info.items()]))

    def __repr__(self):
        return f"{self.name}"

    @property
    def weight_name(self):
        if isinstance(self.__weight,wf.WeightFactory):
            return self.__weight.name
        else:
            # maybe bug appearing
            return str(self.__weight)
    @property
    def name(self):
        if self.__name is None:
            return f"{str(self.__gen_integrand)} - {str(self.__ev)}"
        else:
            return self.__name

    @name.setter
    def name(self,name):
        self.__name = name

    def tick_count(self,x):
        if len(x.shape)==0:
            self.neval +=1
        else:
            self.neval +=x.shape[0]

    def verify_input(self,x):
        x=np.require(x)
        if self.dim==1:
            if len(x.shape)==0:
                x = np.atleast_2d(x)
            elif len(x.shape)==1:
                x = np.atleast_2d(x).reshape(-1,1)
            else:
                raise ValueError(f"Input shape <{x.shape}> does not fit dimension {self.dim}.")
        else:
            if len(x.shape)==1:
                assert x.shape[0]==self.dim, f"Input shape <{x.shape}> does not fit dimension {self.dim}."
                x = np.atleast_2d(x)
            else:
                assert x.shape[-1]==self.dim, f"Input shape <{x.shape}> does not fit dimension {self.dim}."
        return x


    def __integrand(self,x,**params):
        self.tick_count(x)
        return np.prod(self.__gen_integrand(x,**params) - self.__ev(**params),axis = -1)

    def __weighted_integrand(self,x,**params):
        temp_integrand = self.__integrand(x,**params)
        return self.__weight(x)*temp_integrand

    def get_integrand(self,dim,**params):
        #self.params = params
        self.neval = 0
        self.dim = dim
        return lambda x: self.__integrand(self.verify_input(x),**params)

    def get_weighted_integrand(self,dim,**params):
        self.dim = dim
        self.neval = 0
        return lambda x: self.__weighted_integrand(self.verify_input(x),**params)
