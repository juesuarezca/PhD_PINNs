from pkg_resources import get_distribution, DistributionNotFound

try:

    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    version = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from minterpy.multivariate_polynomial_abstract import *
from minterpy.multi_index import *
from minterpy.canonical_polynomial import *
from minterpy.newton_polynomial import *
from minterpy.multi_index_tree import *
from minterpy.grid import *
from minterpy.transformation_abstract import *
from minterpy.transformation_newton import *
from minterpy.lagrange_polynomial import *
from minterpy.transformation_canonical import *
from minterpy.transformation_lagrange import *
from minterpy.regression import *
from minterpy.derivation import *

