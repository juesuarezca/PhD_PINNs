"""
quadpy integrator
"""
import numpy as np
import quadpy

import mintegpy as mt
from mintegpy.diagnostics import count_class

avial_algos = ["dobrodeev_1970",
"dobrodeev_1978",
"ewing",
"hammer_stroud_1n",
"hammer_stroud_2n",
"mustard_lyness_blatt",
"phillips",
"stroud_1957_2",
"stroud_1957_3",
"stroud_1966_a",
"stroud_1966_b",
"stroud_1966_c",
"stroud_1966_d",
"stroud_1968",
"stroud_cn_1_1",
"stroud_cn_1_2",
"stroud_cn_2_1",
"stroud_cn_2_2",
"stroud_cn_3_1",
"stroud_cn_3_2",
"stroud_cn_3_3",
"stroud_cn_3_4",
"stroud_cn_3_5",
"stroud_cn_3_6",
"stroud_cn_5_2",
"stroud_cn_5_3",
"stroud_cn_5_4",
"stroud_cn_5_5",
"stroud_cn_5_6",
"stroud_cn_5_7",
"stroud_cn_5_8",
"stroud_cn_5_9",
"stroud_cn_7_1",
"thacher",
"tyler"]

def integrate_quadpy(func_getter,func_args = None,dim = 2,algo='stroud_cn_3_6'):
    if func_args is None:
        temp_func = func_getter()
    else:
        temp_func = func_getter(**func_args)
    scheme = getattr(quadpy.ncube,algo)(dim)
    @count_class(axis=-1)
    def integrand(x):
        print("x.shape",x.shape)
        return temp_func(x.T)

    val = scheme.integrate(
    integrand,
    quadpy.ncube.ncube_points(*[[0.0, 1.0]]*dim)
    )
    return {'res':max(abs(val),mt.MACHINE_PRECISION),'count':integrand.called}
