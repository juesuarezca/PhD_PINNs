# regression

import time
from typing import Optional, Callable, Union, List

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

from minterpy.global_settings import DEBUG
from minterpy.grid import Grid
from minterpy.joint_polynomial import JointPolynomial
from minterpy.lagrange_polynomial import LagrangePolynomial

__all__ = ['Regression']

from minterpy.utils import report_error
from minterpy.verification import check_type_n_values, check_domain_fit, check_type

WEIGHTED_REGRESSION_MODEL = LinearRegression  # TODO NOTE: non regularised. problematic for underconstrained fits!

# TODO not working with regularised models:
# WEIGHTED_REGRESSION_MODEL = Lasso # l1 regularisation
# WEIGHTED_REGRESSION_MODEL = ElasticNet  # both l1 and l2 regularisation

LAGRANGE_POLY_TYPE = Union[LagrangePolynomial, JointPolynomial]


# NOTE: TODO generally replace all assertions with "raise Error...". only tests should raise AssertionErrors!
# TODO add functionality for adding points without recomputing the whole transformation matrix?
class Regression(object):
    def __init__(self, lagrange_poly: LAGRANGE_POLY_TYPE, verbose: bool = True):
        # def __init__(self, multi_index: Union[MultiIndex, np.ndarray], generating_points=None, verbose: bool = True):
        # if generating_points is not None:
        #     TODO use generating_points, build grid and pass
        #      raise NotImplementedError

        self._regression_matrix: Optional[np.ndarray] = None
        self._sample_points: Optional[np.ndarray] = None
        self._function_values: Optional[np.ndarray] = None
        self._regression_values: Optional[np.ndarray] = None
        self.verbose = verbose

        check_type(lagrange_poly, LAGRANGE_POLY_TYPE.__args__)
        # check_type(lagrange_poly, LAGRANGE_POLY_TYPE)
        self._lagrange_poly: LAGRANGE_POLY_TYPE = lagrange_poly

    @classmethod
    def from_grids(cls, grids: Union[Grid, List[Grid]], verbose: bool = True):
        # NOTE: a list of grids may be gives as input in order compute a fit consisting of different polynomials
        # -> "composite polynomial regression"
        if isinstance(grids, Grid):
            grids = [grids]  # convert into list
        elif type(grids) is not list:
            raise TypeError(f'input must be given as type {Grid} or list of such.')

        lagrange_polys = []
        for grid in grids:
            multi_index = grid.multi_index
            # NOTE: assert that these coefficients will never be used -> initialise to None
            new_poly = LagrangePolynomial(None, multi_index, grid=grid)
            lagrange_polys.append(new_poly)

        if len(lagrange_polys) == 1:  # only one polynomial -> store as regular LagrangePolynomial
            lagrange_poly = lagrange_polys[0]
        else:  # multiple polynomials -> store as joint instance
            lagrange_poly = JointPolynomial(lagrange_polys)
        return cls(lagrange_poly, verbose)

    @property
    def transformation_stored(self):
        return self.sample_points is not None

    # read-only properties:
    @property
    def regression_values(self):
        return self._regression_values

    @property
    def regression_matrix(self):
        return self._regression_matrix

    @property
    def function_values(self):
        return self._function_values

    @property
    def sample_points(self):
        return self._sample_points

    @property
    def error_values(self):
        return self._error_values

    def equal_sample_points_stored(self, sample_points):
        if not self.transformation_stored:
            return False
        if self.sample_points.shape != sample_points.shape:
            return False
        return np.allclose(self.sample_points, sample_points)

    def verify_fct_vals(self, function_values):
        check_type_n_values(function_values)
        # TODO also test numerical properties of the function values to assert numerical stability of regression

    def verify_sample_points(self, sample_points: np.ndarray):
        check_domain_fit(sample_points)  # includes type and value checks
        nr_data_points, m  = sample_points.shape  # ensure that dimensions fit
        if m != self._lagrange_poly.spatial_dimension:
            raise ValueError(
                'the sample points must have the same dimensionality as the regression (polynomials, grids)')
        # TODO sample points must be unique (no point pair too similar, expensive!)
        # TODO test numerical properties -> stability

    def cache_transform(self, sample_points: np.ndarray, verify_input: bool = True):
        if verify_input:
            self.verify_sample_points(sample_points)
        if type(sample_points) is not np.ndarray:
            raise TypeError()

        start_time = time.time()

        # the regression matrix consists of the values of all Lagrange polynomials ("monomials) on all sample points
        # = transformation from the interpolation grid to the data samples(possibly scattered)
        R = self._lagrange_poly.eval_lagrange_monomials_on(sample_points)
        nr_data_samples, m  = sample_points.shape
        nr_coeffs = self._lagrange_poly.nr_active_monomials
        # NOTE: TODO the shape of the evaluation result array is not fixed -> reshape
        R = R.reshape(nr_data_samples, nr_coeffs)
        self._regression_matrix = R
        self._sample_points = sample_points

        if self.verbose:
            fit_time = time.time() - start_time
            print(f'transformation computed in {fit_time:.2e}s')
            cond_nr = np.linalg.cond(R)
            print(f'condition number of the regression matrix: {cond_nr:.2e}')

    def regress_simple(self, function_values):
        # TODO use sklearn Lasso, regularised... regression
        coeffs_lagrange, _, _, _ = scipy.linalg.lstsq(self.regression_matrix, function_values)
        return coeffs_lagrange

    def regress_weighted(self, fct_values: np.ndarray, sample_weights: np.ndarray, verify_input: bool = True):
        if self.verbose:
            print("weighted polynomial regression")
        # raise NotImplementedError
        if verify_input:
            check_type_n_values(sample_weights)
            if fct_values.shape != sample_weights.shape:
                raise ValueError('function values and given weights must possess equal shape, '
                                 f'but {fct_values.shape} != {sample_weights.shape}')
            if np.any(sample_weights < 0.0):
                raise ValueError('all weights must be positive')

        # NOTE: disable the constant term "intercept_"
        regr_obj = WEIGHTED_REGRESSION_MODEL(fit_intercept=False)
        X = self._regression_matrix
        y = fct_values
        regr_obj.fit(X, y, sample_weight=sample_weights)  # , check_input=DEBUG
        # if DEBUG:
        #     predictions = regr_obj.predict(X)
        #     assert np.allclose(predictions, fct_values)  # sanity check
        coeffs_lagrange = regr_obj.coef_
        return coeffs_lagrange

    def _regr_wrapper(self, core_regression_fct: Callable,
                      function_values: np.ndarray, sample_points: Optional[np.ndarray] = None,
                      use_cached_transform: bool = False, verify_input: bool = True,
                      *args, **kwargs):
        """ defines equal behaviour of all regression fcts      """

        if verify_input:
            self.verify_fct_vals(function_values)

        if use_cached_transform:
            if not self.transformation_stored:
                raise ValueError('trying to use cached transformation, but none as been stored yet.')
        else:
            if sample_points is None:
                raise ValueError('sample points must be given in order to compute a transformation')
                # without regression without an available transformation')
            self.cache_transform(sample_points, verify_input)  # independent of the function values

        start_time = time.time()
        # independent on the sample points (transformation computed earlier)
        # the Lagrange coeffs (=values) of the polynomial on the interpolation nodes (grid)
        coeffs_lagrange = core_regression_fct(function_values, *args, **kwargs)
        if self.verbose:
            fit_time = time.time() - start_time
            print(f'fit took {fit_time:.2e}s')

        self._function_values = function_values
        self._lagrange_poly.coeffs = coeffs_lagrange
        self._regression_values = self._regression_matrix @ coeffs_lagrange
        if DEBUG:  # sanity check
            assert len(self._regression_values) == len(function_values)
        self._error_values = self._regression_values - function_values
        if self.verbose:
            report_error(self.error_values, description="errors on the data samples:")

        return self._lagrange_poly

    # TODO flip arguments, functions should have identical argument ordering
    def regression(self, sample_points: Optional[np.ndarray], function_values: np.ndarray,
                   use_cached_transform: bool = False, verify_input: bool = True):
        """ fits polynomial using simple regression approach.

        make sure that the points `grid_points` are normalized within range [-1,1]
        """
        return self._regr_wrapper(self.regress_simple,
                                  function_values=function_values,
                                  sample_points=sample_points,
                                  use_cached_transform=use_cached_transform,
                                  verify_input=verify_input)

    # TODO flip arguments, functions should have identical argument ordering
    def weighted_regression(self, sample_points: Optional[np.ndarray], function_values: np.ndarray,
                            sample_weights: np.ndarray,
                            use_cached_transform: bool = False, verify_input: bool = True):
        """ fits polynomial using weighted regression approach.

        make sure that the points `grid_points` are normalized within range [-1,1]
        """
        return self._regr_wrapper(self.regress_weighted,
                                  function_values=function_values,
                                  sample_points=sample_points,
                                  use_cached_transform=use_cached_transform,
                                  verify_input=verify_input,
                                  sample_weights=sample_weights)
