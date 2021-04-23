import os
import pathlib
import time

import numpy as np

import mintegpy.minterpy.utils as utils
from mintegpy.Integrator import integration_points_weights

current_dir = '.'

def get_points_weights(input_para,point_kind,use_data=False,optimal= False):
    generator= integration_points_weights(*input_para, use_data=use_data,point_kind=point_kind,optimal = optimal)
    return np.vstack((generator.PP,generator.Li))


def build_points_weights_data(input_para,point_kind='leja',file_sufix = "points_weights",file_prefix = "csv",data_dir = ".", printopt=False,use_data=False,optimal=False):
    start = time.time()
    out = get_points_weights(input_para,point_kind,use_data=use_data,optimal=optimal)
    end = time.time() - start


    if optimal:
        opt = "_opt"
    else:
        opt= ""
    filename = "%s%s_m%dn%dl%d.%s"%(file_sufix,opt,input_para[0],input_para[1],input_para[2],file_prefix)
    save_file = os.path.join(current_dir,data_dir,filename)
    head =" ".join(["x%d"%d for d in np.arange(1,input_para[0]+1)] + ["Li"])
    foot = "dim = %d, polydegree = %d, lp_degree = %d, point_kind = %s, optimal = %s"%(input_para[0],input_para[1],input_para[2],point_kind,str(optimal))

    np.savetxt(save_file, out.T, delimiter=" ",header=head,footer = foot)


    if printopt:
        print("="*20)
        print("input (dim,polydeg,lp)",input_para)
        print("point kind",point_kind)
        print("PP shape",out[:-1].shape)
        print("Li shape",out[-1].shape)
        print("build time %1.2e"%(end))
        print("Saved in <%s>"%save_file)
        print("="*20)

    return 0
