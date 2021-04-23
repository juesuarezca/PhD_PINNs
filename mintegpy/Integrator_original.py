__all__ = ['integrate']
import numpy as np
from scipy import special

from mintegpy.DataHandler import reader
from mintegpy.minterpy.Transform import transform, transform_optimal
from mintegpy.minterpy.utils import choose_right_points_for_hybrid

def monomial_integrate(gamma,weight_function = 'legendre'):
    #print("weight_function",weight_function)
    if weight_function=='legendre':
        return monomial_integrate_leg(gamma)
    elif weight_function=='cheb1':
        return monomial_integral_cheb1st(gamma)
    elif weight_function=='cheb2':
        return monomial_integral_cheb2nd(gamma)
    else:
        raise ValueError("There is no weight function called <%>"%weight_function)

def monomial_integrate_leg(gamma):
    #print("Note: gauss legendre weights used.")
    #print("!!!! Warning: integration on odd monomials are set zero... check if this is right!")
    case = np.mod(gamma,2)==0
    temp_gamma = gamma[case]
    temp_integ = 2/(temp_gamma+1)
    res = np.zeros(gamma.shape)
    res[case] = temp_integ
    return res.prod(axis = 0)

def monomial_integral_cheb1st(exponents):
    #print("Note: chebychev 1st weights are used: 1/sqrt(1-x^2)")
    #print("monomial integral exponents",exponents)
    cond = exponents>0
    term = np.ones(exponents.shape)*np.pi
    term[cond] = np.sqrt(np.pi)/exponents[cond] * ((-1)**exponents[cond] + 1)*special.gamma((exponents[cond] + 1)/2)/special.gamma(exponents[cond]/2)
    return np.prod(term,axis = 0)

def monomial_integral_cheb2nd(exponents):
    #print("Note: chebychev 2nd weights are used: sqrt(1-x^2)")
    term = np.sqrt(np.pi)/4*((-1)**exponents + 1)*special.gamma((exponents + 1)/2)/special.gamma(exponents/2+2)
    return np.prod(term, axis = 0)

class integration_points_weights(transform):
    def __init__(self,M,n,lp,use_data,point_kind,optimal = False,split = None,weight_function = 'legendre'):
        self.M = M
        self.n = n
        self.lp = lp
        self.point_kind = point_kind
        self.__data_success = False
        self.weight_function = weight_function
        if use_data:
            # TODO: weight-function is not a flag for data load!!
            points_weights = reader(M,n,lp,kind = point_kind,file_sufix=point_kind,optimal=optimal)
            self.__data_success = (points_weights is not None)
            print("data?",self.__data_success)
            if self.__data_success:
                self.__PP = points_weights[:-1]
                self.__integ_li = points_weights[-1]
        if not (self.__data_success and use_data):
            #print("Warning: the is no data for dim: %d, polydeg: %d, lp: %d. -> fallback to recalc. "%(M,n,lp))
            transform.__init__(self,M,n,lp,point_kind,split = split)
            self.__PP = self.tree.PP
            #print("point shape",self.__PP.shape)
            if 'hybrid' in self.point_kind:
                self.cond = choose_right_points_for_hybrid(self.tree.Gamma,self.tree.split)
                #print("hybrid condition triggered.")
            else:
                self.cond = np.ones(self.__PP[0].shape,dtype=np.bool)

            self.__PP = self.__PP[:,self.cond]
            #print("point shape",self.__PP.shape)
            self.__build_trans_prod()

        if optimal and not(self.__data_success):
            #print("Note: optimal transformation used.")
            #print("calc optimal transformation for dim: %d, polydeg: %d, lp: %d."%(M,n,lp))
            self.__transform_to_optimal()
        else:
            self.__PP_out = self.__PP
            self.__integ_li_out = self.__integ_li

    def __build_trans_prod(self):
        self.__integ_li = np.dot(self.trans_matrix.T,monomial_integrate(self.trans_gamma,self.weight_function))[self.cond]

    def __transform_to_optimal(self):
        #print("trigger optimal transform")
        self.__trans_optimal = transform_optimal(M=self.M, n=self.n, lp=self.lp,point_kind=self.point_kind)
        self.__PP_out = self.__trans_optimal.tree.optimalPP
        self.__integ_li_out = self.__trans_optimal.transform_weights(self.__integ_li)


    @property
    def PP(self):
        return self.__PP_out

    @property
    def Li(self):
        return self.__integ_li_out
"""

class integration_points_weights(object):
    def __init__(self,M,n,lp):
        self.__data_success = False
        points_weights = reader(M,n,lp)
        self.__data_success = (points_weights is not None)
        if self.__data_success:
            self.__PP = points_weights[:-1]
            self.__integ_li = points_weights[-1]
        elif not (self.__data_success):
            raise ValueError("There is no data for dim: %d, polydeg: %d, lp: %d yet."%(M,n,lp))

    @property
    def PP(self):
        return self.__PP

    @property
    def Li(self):
        return self.__integ_li"""

class default_integrate(integration_points_weights):
    def __init__(self,M=2,n=2,lp=2,use_data=True,point_kind='leja',optimal = False,split = None,weight_function='legendre'):
        integration_points_weights.__init__(self,M,n,lp,use_data,point_kind,optimal,split = split,weight_function = weight_function)

    def integrate(self,func):
        return np.dot(func(self.PP),self.Li)


class integrate(default_integrate):
    def __init__(self,M,n,lp,a=-1,b=1,use_data=True,point_kind='leja',optimal=False,split = None,weight_function='legendre'):
        #print(f"Note: input point_kind {point_kind}")
        default_integrate.__init__(self,M,n,lp,use_data,point_kind,optimal,split = split,weight_function = weight_function)
        self.__dim = M
        if np.isscalar(a):
            self.__a = np.ones(self.__dim)*a
        elif len(a) == self.__dim:
            self.__a = a
        else:
            raise ValueError("Shape of a needs to be %s (%s given)"%(str(self.__dim),str(a.shape)))

        if np.isscalar(b):
            self.__b = np.ones(self.__dim)*b
        elif len(b) == self.__dim:
            self.__b = b
        else:
            raise ValueError("Shape of b needs to be %s (%s given)"%(str(self.__dim),str(b.shape)))
        self.__slope = (self.__b - self.__a)/2
        #print("self.__slope",self.__slope)
        self.__shift = (self.__a + self.__b)/2
        #print("self.__shift",self.__shift)

    @property
    def slope(self):
        return self.__slope

    @property
    def shift(self):
        return self.__shift
    @property
    def points(self):
        #print("self.PP",self.PP.shape)
        temp_pts = self.slope[:,np.newaxis]*self.PP + self.shift[:,np.newaxis]
        return temp_pts

    @property
    def weights(self):
        #print("self.Li",self.Li.shape)
        #print("self.slope",self.slope)
        #print("np.prod(self.slope)",np.prod(self.slope))
        return np.prod(self.slope) * self.Li

    def integrate(self,func):
        #cond points and weights -> if point_kind is hybrid

        #print("point_shape",self.points.shape)
        self.fkt_vals = func(self.points.T)
        return np.dot(self.fkt_vals,self.weights)

if __name__ == '__main__':
    from diagnostics import count
    from scipy.integrate import dblquad,tplquad
    from otherIntegrators import ngauss_quad

    input_para = (3,5,1)
    test_integrate1 = integrate(*input_para,use_data=True)



    @count
    def f(x):
        return -2*x[0]**4 + 4*x[1]**2 + 3*x[2]*x[0]**4 + 10*x[1]**4

    def f_wrapper(x,y,z):
        return f([x,y,z])

    test_res = test_integrate1.integrate(f)

    count_ours = f.called


    test_res_quad = tplquad(f_wrapper, -1, 1, lambda x: -1, lambda x: 1,lambda x, y: -1, lambda x, y: 1)
    count_quad = f.called - count_ours

    print("==== COMPARE TO SCIP.TPLQUAD ====")
    print("---- setting ----")
    print("dim: %d, deg: %d, lp: %d"%(input_para[0],input_para[1],input_para[2]))
    print()

    print("---- result -----")
    print("our result",test_res)
    print("quad result",test_res_quad)
    print("abs_err",np.abs(test_res-test_res_quad[0]))
    print()

    print("---- function calls ----")
    print("func calls (our)",count_ours)
    print("func calls (quad)",count_quad)
    print()

    input_para = (4,7,1)
    test_integrate = integrate(*input_para)

    test_gauss = ngauss_quad(input_para[0],input_para[1])

    @count
    def f(x):
        return -2*x[0]**4 + 4*x[1]**2 + 3*x[2] + 10 + 11*x[2]**6 + 100*x[3]**5*x[2]**2 -89*x[1]**7


    test_res = test_integrate.integrate(f)

    count_ours = f.called


    test_res_gauss = test_gauss.integrate(f)
    count_gauss = f.called - count_ours

    print("==== COMPARE TO GAUSS ====")
    print("---- setting ----")
    print("dim: %d, deg: %d, lp: %d"%(input_para[0],input_para[1],input_para[2]))
    print()

    print("---- result -----")
    print("our result",test_res)
    print("gauss result",test_res_gauss)
    print("abs_err",np.abs(test_res-test_res_gauss))
    print()

    print("---- function calls ----")
    print("func calls (our)",count_ours)
    print("func calls (gauss)",count_gauss)
