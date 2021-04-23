# -*- coding:utf-8 -*-
"""
general TODO:
 - build leja ordering function for input pointset
"""
import numpy as np
from scipy import special


def gauss_leg_values(n):
    #print("Note: gauss_leg_values used.")
    if n>=100:
        print("Warning: The used numpy implementation becomes unstable for degree larger that n = 100. (<%s> given)"%n)
    points, _ = np.polynomial.legendre.leggauss(n+1)
    return points[None,:]


def chebpoints_2nd_order(n):  # 2nd order
    #print("Note: chebychev 2nd used.")
    if n==1:
        return np.array([1.0])
    return np.cos(np.arange(n, dtype=np.float128) * np.pi / (n - 1))


def chebpoints_2nd_order_unbound(n):  # 2nd order
    return special.roots_chebyu(n+1)[0]


def chebpoints_1st_order(n):  # 2nd order
    #print("Note: chebychev 1st used.")
    if n==1:
        return np.array([0.0])
    K = np.arange(1,n+1,dtype=np.float128)
    return np.cos((2*K -1) * np.pi /( 2*(n + 1)))

def chebpoints_1st_order_unbound(n):  # 2nd order
    #print("Note: chebychev 1st unbound used.")
    return np.polynomial.chebyshev.chebgauss(n+1)[0]

def leja_ordered_values(n):
    #print("Note: chebychev 2nd leja ordered used.")
    #print("leja cheb call: n=",n)
    #print("cheb preorder",chebpoints_2nd_order(n + 1))
    points1 = chebpoints_2nd_order(n + 1)[::-1]

    #print("cheb points",points1)
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points

def leja_ordered_values_unbound(n):
    #print("Note: chebychev 2nd unbound leja ordered used.")
    #print("leja cheb call: n=",n)
    #print("cheb preorder",chebpoints_2nd_order(n + 1))
    points1 = chebpoints_2nd_order_unbound(n)[::-1]
    #points1 = np.delete(points1,(0,-1)) # avoid bounds

    #print("cheb points",points1)
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points

def chebpoints_2nd_order_split(n):
    #print("Note: chebychev 2nd split used.")
    split = int(n/2)
    small = leja_ordered_values(split)
    big = chebpoints_2nd_order(2*split+1)[1::2]
    merged = np.concatenate((small,big.reshape(1,len(big))),axis=-1)
    print("merged.shape",merged.shape)
    return merged


def leja_ordered_cheb_1st(n):
    #print("Note: chebychev 1st leja ordered used.")
    #print("leja cheb call: n=",n)
    #print("cheb preorder",chebpoints_2nd_order(n + 1))
    points1 = chebpoints_1st_order(n + 1)[::-1]

    #print("cheb points",points1)
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points

def leja_ordered_cheb_1st_unbound(n):
    #print("Note: chebychev 1st leja ordered used.")
    #print("leja cheb call: n=",n)
    #print("cheb preorder",chebpoints_2nd_order(n + 1))
    points1 = chebpoints_1st_order_unbound(n)[::-1]

    #print("cheb points",points1)
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points

def leja_ordered_gauss(n):
    #print("Note: gauss_leg leja ordered used.")
    #print("leja gauss call: n=",n)
    points, _ = np.polynomial.legendre.leggauss(n+1)
    points1 =points[::-1]
    #print("gauss shape",points1.shape)
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if (P >= m):
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([1, n + 1])
    for i in range(n + 1):
        leja_points[0, i] = points2[int(lj[0, i])]
    return leja_points


def gauss_cheb_hybrid(n,split = None):
    """
    generates 'n' hybrid points splitted at 'split' into gauss and cheb, resp. (both leja ordered)

    Parameters
    ==========
    n int64
        number of points

    split int64
        point of splitting between gauss and cheb (0<k<n)
    """
    #print("Note: hybrid gauss_leg leja ordered/chebychev 2nd leja ordered used.")

    if split is None:
        raise ValueError("There is no 'split' for hybrid points!")
    #print("leja gauss call: n=",n)

    pointsGauss = leja_ordered_gauss(split)
    #print("pointsGauss",pointsGauss)
    if (n-split)%2==1 and split%2==0:
        pointsCheb = leja_ordered_values(n-split)
        #print("pointsCheb",pointsCheb)
        #print("pointsCheb used",pointsCheb[:,:(n-split-1)])
        points = np.concatenate((pointsGauss,pointsCheb[:,:(n-split)]),axis=-1)
    else:
        pointsCheb = leja_ordered_values(n-split-1)
        points = np.concatenate((pointsGauss,pointsCheb),axis = -1)

    #print("pointsGauss",pointsGauss.shape)
    #print("pointsCheb",pointsCheb.shape)
    #print("points",points)
    return points

def gauss_gauss_hybrid(n,split = None):
    """
    generates 'n' hybrid points splitted at 'split' into gauss and cheb, resp. (both leja ordered)

    Parameters
    ==========
    n int64
        number of points

    split int64
        point of splitting between gauss and cheb (0<k<n)
    """
    #print("Note: hybrid gauss_leg leja ordered/gauss_leg leja ordered used.")
    #print("hybrid n",n)
    #print("hybrid split",split)

    if split is None:
        raise ValueError("There is no 'split' for hybrid points!")
    #print("leja gauss call: n=",n)

    pointsGauss = leja_ordered_gauss(split)
    #print("pointsGauss",pointsGauss)
    if (n-split)%2==1 and split%2==0:
        pointsGauss2 = leja_ordered_gauss(n-split)
        #print("pointsCheb",pointsCheb)
        #print("pointsCheb used",pointsCheb[:,:(n-split-1)])
        points = np.concatenate((pointsGauss,pointsGauss2[:,:(n-split)]),axis=-1)
    else:
        pointsGauss2 = leja_ordered_gauss(n-split-1)
        points = np.concatenate((pointsGauss,pointsGauss2),axis = -1)

    #print("points",points)
    return points

def gauss_cheb1_hybrid(n,split = None):
    """
    generates 'n' hybrid points splitted at 'split' into gauss and cheb, resp. (both leja ordered)

    Parameters
    ==========
    n int64
        number of points

    split int64
        point of splitting between gauss and cheb (0<k<n)
    """
    #print("Note: hybrid gauss_leg leja ordered/chebychev 1st leja ordered used.")
    #print("hybrid n",n)
    #print("hybrid split",split)

    if split is None:
        raise ValueError("There is no 'split' for hybrid points!")
    #print("leja gauss call: n=",n)

    pointsGauss = leja_ordered_gauss(split)
    #print("pointsGauss",pointsGauss)
    if (n-split)%2==1 and split%2==0:
        pointsCheb1 = leja_ordered_cheb_1st(n-split)
        #print("pointsCheb",pointsCheb)
        #print("pointsCheb used",pointsCheb[:,:(n-split-1)])
        points = np.concatenate((pointsGauss,pointsCheb1[:,:(n-split)]),axis=-1)
    else:
        pointsCheb1 = leja_ordered_cheb_1st(n-split-1)
        points = np.concatenate((pointsGauss,pointsCheb1),axis = -1)

    #print("points",points)
    return points

__avial_generator = {
    'cheb1':lambda n: chebpoints_1st_order(n+1),
    'cheb1unbound':chebpoints_1st_order_unbound,
    'cheb2':lambda n: chebpoints_2nd_order(n+1),
    'cheb2_unbound':chebpoints_2nd_order_unbound,
    'leja_cheb2_unbound':leja_ordered_values_unbound,
    'leja_cheb1_unbound':leja_ordered_cheb_1st_unbound,
    'cheb2_split':chebpoints_2nd_order_split,
    'leja':leja_ordered_values,
    'leja_1st':leja_ordered_cheb_1st,
    'gauss_leg': gauss_leg_values,
    'leja_gauss': leja_ordered_gauss,
    'hybrid-cheb2':gauss_cheb_hybrid,
    'hybrid-cheb1':gauss_cheb1_hybrid,
    'hybrid-gauss':gauss_gauss_hybrid
}


def generate_points(n,kind=None,dim=None,split=None):
    """
    generates the interpolation points of a given kind and polynom degree n.
    """
    if kind is None:
        return __avial_generator['leja'](n)
    elif kind in __avial_generator.keys():
        # ToDo: harmonize point generator input
        if 'hybrid' in kind:
            #print("split",split)
            return __avial_generator[kind](n,split)
        else:
            return __avial_generator[kind](n)
    else:
        raise ValueError("There is no pointset named <%s>!\nAvailable pointsets are %s"%(kind,list(__avial_generator.keys())))



# multi indices gamma
def gamma_lp(m, n, gamma, gamma2, p):
    # TODO better names
    gamma0 = gamma.copy()
    gamma0[m - 1] = gamma0[m - 1] + 1

    out = []
    norm = np.linalg.norm(gamma0.reshape(-1), p)
    if norm < n and m > 1:
        o1 = gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), p)
        o2 = gamma_lp(m, n, gamma0.copy(), gamma0.copy(), p)
        out = np.concatenate([o1, o2], axis=-1)
    elif norm < n and m == 1:
        out = np.concatenate([gamma2, gamma_lp(m, n, gamma0.copy(), gamma0.copy(), p)], axis=-1)
    elif norm == n and m > 1:
        out = np.concatenate([gamma_lp(m - 1, n, gamma.copy(), gamma.copy(), p), gamma0], axis=-1)
    elif norm == n and m == 1:
        out = np.concatenate([gamma2, gamma0], axis=-1)
    elif norm > n:
        norm_ = np.linalg.norm(gamma.reshape(-1), p)
        if norm_ < n and m > 1:
            for j in range(1, m):
                gamma0 = gamma.copy()
                gamma0[j - 1] = gamma0[j - 1] + 1  # gamm0 -> 1121 broken
                if np.linalg.norm(gamma0.reshape(-1), p) <= n:
                    gamma2 = np.concatenate([gamma2, gamma_lp(j, n, gamma0.copy(), gamma0.copy(), p)], axis=-1)
            out = gamma2
        elif m == 1:
            out = gamma2
        elif norm_ <= n:
            out = gamma

    return out


# TODO restructure code: should be accessible as simple function eval_poly(x)
def get_eval_fct(tree, coeffs_newton, m, n, N, gamma):
    # TODO copy needed?!
    # TODO all arguments needed? 1, 1?!
    return lambda x: tree.eval_lp(x.copy(), coeffs_newton.copy(), m, n, N, gamma.copy(),
                                  tree.GP.copy(), tree.lpDegree, 1, 1)


def apply_vectorized(eval_fct, eval_points):
    return np.apply_along_axis(eval_fct, 0, eval_points)


def report_on_error(description, res):
    print(f"\n{description}\n"
          f"average: {np.ma.average(res)}\n"
          f"l_2 error: {np.linalg.norm(res)}\n"
          f"l_infty error (max): {np.linalg.norm(res, ord=np.inf)}\n")


def eval_fct_canonical(x, coefficients, exponents):
    coeffs_copy = coefficients.copy()
    nr_coeffs = len(coefficients)
    nr_dims, nr_monomials = exponents.shape
    assert nr_monomials == nr_coeffs
    assert len(x) == nr_dims
    for i in range(nr_coeffs):
        coeffs_copy[i] = coeffs_copy[i] * np.prod(np.power(x, exponents[:, i]))

    return np.sum(coeffs_copy)


def get_eval_fct_canonical(coefficients, exponents):
    # return lambda x: np.sum(coefficients.T * np.prod(np.power(x, exponents), axis=1), axis=1)[0]
    return lambda x: eval_fct_canonical(x, coefficients, exponents)


def choose_right_points_for_hybrid(gamma,k):
    max_alpha = np.max(gamma, axis = 0)
    #print("max_alpha",max_alpha)
    cond = np.logical_and(max_alpha>k,max_alpha<=(2*k+1))
    return cond


# TODO is this a test, routine...?
if __name__ == '__main__':
    pass
    """
    from scipy.special import roots_chebyt, roots_chebyu
    import scipy  # required where?

    print("scipy version", scipy.__version__)
    import matplotlib.pylab as plt  # TODO add to project requirements?

    import h5py  # TODO add to project requirements?

    with h5py.File("chebpts.mat", 'r') as chebpts:
        cp_10 = np.asarray(chebpts['cp_10'])[0]
        cp_50 = np.asarray(chebpts['cp_50'])[0]
        cp_100 = np.asarray(chebpts['cp_100'])[0]
        cp_500 = np.asarray(chebpts['cp_500'])[0]
        cp_1000 = np.asarray(chebpts['cp_1000'])[0]
        chebfun_points = [cp_10, cp_50, cp_100, cp_500, cp_1000]

    n_arr = np.array([10, 50, 100, 500, 1000])
    for i, n in enumerate(n_arr):
        a = chebfun_points[i]
        b = chebpoints_2nd_order(n)[::-1]  # roots_chebyt(n)[0]
        # b=roots_chebyu(n)[0]
        abs_err = np.abs(a - b)
        rel_err = abs_err / (np.abs(a + b))
        print('mean abs err', abs_err.mean())
        print('mean rel err', rel_err.mean())
        plt.plot(n, abs_err.mean(), 'or')
        plt.plot(n, rel_err.mean(), 'Xk')
        plt.plot(n, abs_err.max(), '>g')

    plt.yscale('log')
    plt.show()
    """
