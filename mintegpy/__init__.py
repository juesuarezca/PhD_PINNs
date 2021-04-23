# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution
"""
top level integration
"""
from mintegpy.datapaths import *
import mintegpy.minterpy.utils as utils
import mintegpy.PointsWeightsGenerator as PWgen
from mintegpy.DataHandler import *
from mintegpy.Integrator import *
#from mintegpy.benchmark import *
import mintegpy.otherIntegrators as otherIntegrators
import mintegpy.diagnostics as diagnostics
import mintegpy.helper as helper
import mintegpy.integ_test_toolbox as integ_test_toolbox

import mintegpy.benchmark as benchmark


"""
global constants
"""
MACHINE_PRECISION =  7./3 - 4./3 -1


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
