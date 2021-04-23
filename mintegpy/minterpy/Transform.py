# -*- coding:utf-8 -*-
import numpy as np
from scipy.linalg import solve_triangular

from mintegpy.minterpy.Solvers import interpol
from mintegpy.minterpy.utils import gamma_lp

#from mintegpy.minterpy.diagnostics import TIMES, timer


# TODO define slots for all classes


# TODO rename
# TODO again child? hierarchy really needed?
class transform(interpol):
    def __init__(self, M=2, n=2, lp=2,point_kind='leja',split = None):
        #print("split transform",split)
        interpol.__init__(self, M, n, lp,point_kind,split=split)
        self.__build_vandermonde_n2c()
        self.__build_transform_n2c()
        self.__build_vandermonde_l2n()
        self.__build_transform_l2n()
        self.__build_trans_Matrix()

    def __build_vandermonde_n2c(self):
        self.init_gamma = np.zeros((self.M, 1))
        self.trans_gamma = (gamma_lp(self.M, self.n, self.init_gamma, self.init_gamma.copy(), self.lp))
        self.V_n2c = np.ones((self.N, self.N))
        for i in np.arange(0, self.N):
            for j in np.arange(1, self.N):
                for d in np.arange(0, self.M):
                    self.V_n2c[i, j] = self.V_n2c[i, j] * self.tree.PP[d, i] ** self.trans_gamma[d, j]

    def __build_transform_n2c(self):
        self.Cmn_n2c = np.zeros((self.N, self.N))
        for j in np.arange(self.N):
            self.Cmn_n2c[:, j] = self.run(self.M, self.N, self.tree.tree, self.V_n2c[:, j].copy(), self.tree.GP.copy(),
                                          self.gamma.copy(), 1, 1)
        self.inv_Cmn_n2c = solve_triangular(self.Cmn_n2c, np.identity(self.N))

    def transform_n2c(self, d):
        return np.dot(self.inv_Cmn_n2c, d)

    def __build_vandermonde_l2n(self):
        self.V_l2n = np.eye(self.V_n2c.shape[0])

    def __build_transform_l2n(self):
        self.Cmn_l2n = np.zeros((self.N, self.N))
        for j in np.arange(self.N):
            self.Cmn_l2n[:, j] = self.run(self.M, self.N, self.tree.tree, self.V_l2n[:, j].copy(), self.tree.GP.copy(),
                                          self.gamma.copy(), 1, 1)

    def transform_l2n(self, l):
        # return solve_triangular(self.Cmn_l2n,l)
        return np.dot(self.Cmn_l2n, l)

    def __build_trans_Matrix(self):
        # in Cmn_l2n: set rows and columns zero if cond(gamma) is not fulfilled!
        self.trans_matrix = np.dot(self.inv_Cmn_n2c, self.Cmn_l2n)

    def transform_l2c(self, v):
        # return self.transform_n2c(self.transform_l2n(v))
        return np.dot(self.trans_matrix, v)

    # TODO static, no need to define this as a class function
    def buildCanonicalVandermonde(self, X, gamma):
        """
        Canonical matrix for polynomial interpolation in normal form
        Y = a * V
        a.. coefficients
        V.. vandermonde matrix
        Y.. interpolated signal
        Example:
            x_in: input vector
            Y_hat = torch.mm(canon_coefs,
                    buildCanonicalVandermonde(x_in, gamma).t())

        Parameters:
        X.. X positions
        gamma.. gamma vectors of our polynomial
        """
        noX, dimX = X.shape
        dimGamma, noCoefficients = gamma.shape
        assert dimX == dimGamma, "Input dimensions (%d,%d) of (X, gamma) dont match." % (dimX, dimGamma)
        V = np.ones((noX, noCoefficients))
        # TODO: this probably only works for m == 2
        for j in range(noCoefficients):
            for k in range(dimX):
                V[:, j] *= (X[:, k] ** gamma[k, j])
            # V[:,j] = (X[:,0]**gamma[0,j]) * (X[:,1]**gamma[1,j])
        return V


class transform_optimal(interpol):
    """
    refac: avoid code repetition
    """
    def __init__(self, M=2, n=2, lp=2,point_kind='leja',split = None):
        interpol.__init__(self, M, n, lp,point_kind,optimal=True,split = split)
        self.__build_vandermonde_l2n()
        self.__build_transform_l2n()
        self.__build_trans_r2u()
        self.__build_trans_u2r()


    def __build_vandermonde_l2n(self):
        self.V_l2n = np.eye(self.N)

    def __build_transform_l2n(self):
        self.Cmn_l2n = np.zeros((self.N, self.N))
        for j in np.arange(self.N):
            self.Cmn_l2n[:, j] = self.run(self.M, self.N, self.tree.tree, self.V_l2n[:, j].copy(), self.tree.GP.copy(),
                                          self.gamma.copy(), 1, 1)

    def __build_trans_r2u(self):
        e_beta = np.zeros(self.N)
        R = np.zeros([self.N, self.N])
        for j in range(self.N):
            D_alpha = self.Cmn_l2n[:, j]
            for i in range(self.N):
                e_beta[i] = self.tree.eval_lp(self.tree.optimalPP[:, i].copy(), D_alpha.copy(), self.M, self.n, self.N, self.gamma.copy(),self.tree.GP.copy(), self.lp, 1, 1)
            R[:, j] = e_beta

        self.Tmn_r2u = R


    def __build_trans_u2r(self):
        #proberly use a faster algorithm for invertation
        self.Tmn_u2r = np.linalg.inv(self.Tmn_r2u)

    def transform_weights(self,weights):
        return np.dot(self.Tmn_u2r.T,weights)
