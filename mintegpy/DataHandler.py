"""
interface for points and weights data
"""
__all__ = ['reader']
import os

import numpy as np

import mintegpy as mint

points_set = {
    'leja':'chebyshev_2nd',
    'gauss_leg':'gauss_leg',
    'leja_gauss':'leja_gauss'
            }


def reader(dim,polydeg,lp,kind = 'leja',file_sufix = 'points_weights',file_prefix = "csv",optimal=False):
    print("optimal",optimal)
    if optimal:
        opt = "_opt"
    else:
        opt= ""

    print("opt",opt)
    if kind not in points_set.keys():
        raise ValueError("There is no kind of points called <%s>!\n Available options are %s"%(kind,list(points_set.keys())))
    data_dir = mint.datapath.data_points_weights
    filename = "%s%s_m%dn%dl%d.%s"%(file_sufix,opt,dim,polydeg,lp,file_prefix)

    load_file = os.path.join(data_dir,points_set[kind]+opt,filename)
    print("try data from: \n%s"%load_file)
    try:
        data = np.genfromtxt(load_file)
        print("loaded data from: \n%s"%load_file)
        return data.T
    except:
        return None


if __name__=='__main__':
    print(reader(2,3,1))
    print(reader(2,2,1,kind='test'))
