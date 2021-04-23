"""
NGAUSSIAN
"""
import numpy as np
from scipy.special import erf

def random_para(dim,ngauss,alpha_lim=[2,10],beta_lim=[0,1],seed=12345678):
    alphas = np.round(np.random.uniform(*alpha_lim,(dim,ngauss)),2)
    betas = np.round(np.random.uniform(*beta_lim,(dim,ngauss)),2)
    return (alphas,betas)

def cumGauss(x):
    return .5 + .5*erf(x/np.sqrt(2))

def get_ngaussian(alpha,beta):
    print("intern alpha",alpha.shape)
    print("intern beta",beta.shape)
    # alpha.shape = (d,n)
    # beta.shape = (d,n)
    #exp_fac = np.sqrt(np.pi)/alpha*(erf(alpha*beta) + erf(alpha*(1-beta)))
    #e = np.sum(np.prod(exp_fac,axis=0))
    r = cumGauss((1-beta)*np.sqrt(2)*alpha)
    t = cumGauss(-beta*np.sqrt(2)*alpha)
    e = np.sum(np.prod(np.sqrt(np.pi)/alpha*(r-t),axis=0))
    def ngaussian(x):
        print("intern x",x.shape)
        fac = np.exp(-np.sum(alpha[None,:,:]**2*(x[:,:,None] - beta[None,:,:])**2,axis=1))
        #print(fac.shape)
        return np.sum(fac,axis=-1) - e
    return ngaussian
