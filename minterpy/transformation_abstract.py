"""
Submodule for the transformation class
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from minterpy.grid import Grid
from minterpy.multi_index import MultiIndex
from minterpy.multivariate_polynomial_abstract import MultivariatePolynomialSingleABC
from minterpy.transformation_operator_abstract import TransformationOperatorABC

__all__ = ['TransformationABC']


class TransformationABC(ABC):
    """
    Abstract base class for Transformation class.
    Stores the transformation matrix and associated metadata.
    """

    available_transforms = {}

    # TODO remove generating points as input. not required! stored in polynomial!
    def __init__(self, origin_poly: MultivariatePolynomialSingleABC, generating_points: Optional[np.ndarray] = None):
        if not isinstance(origin_poly, MultivariatePolynomialSingleABC):
            raise TypeError(f"<{origin_poly}> is not a Polynomial type.")
        if not isinstance(origin_poly, self.origin_type):
            raise TypeError(f"<{origin_poly}> is not of the expected type {self.origin_type}.")
        self.multi_index: MultiIndex = origin_poly.multi_index
        self.origin_poly = origin_poly


        # TODO check for index completeness
        # is complete: -> store the transformation fct once!?
        # TODO transformation fct as an attribute -> assign here (e.g. barycentric if complete)

        # TODO automatic make complete?
        # self.origin_poly = self.origin_poly.make_complete()
        # raise ValueError('some transformations only work for complete multi index sets!')

        # TODO Check if it is correct to use the default grid in all cases.
        if origin_poly.grid is None:
            # The Canonical Polynomial has no grid. generate the default grid
            # TODO attention. is this intuitive for the user?
            #  if this is the default, could add it to the polynomial in the first place!
            self.grid = Grid(self.multi_index)
        else:
            self.grid = origin_poly.grid

        # TODO remove?!
        if generating_points is not None:
            raise NotImplementedError(
                'the generating points should not be passed as input. should be stored in origin polynomial')
        # self.generating_points = generating_points
        self._transformation_operator: Optional[np.ndarray] = None

    # TODO register the transformation classes to the available_transforms dictionary
    # TODO integrate function to retrieve the proper transformation (cf. transformation_utils.py)
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.available_transforms[cls._short_name] = cls

    # TODO: remove argument. store origin poly once and reuse.
    #  otherwise the user could input an incompatible polynomial
    #  avoids ugly duplicate calls:
    #   l2n_transformation = TransformationLagrangeToNewton(lagrange_poly)
    #   newton_poly = l2n_transformation(lagrange_poly)
    # on the other hand avoid constructing multiple transformation objects for the same basis
    # just for transforming multiple different polynomioals! -> rather check the validity of the input polynomial basis
    def __call__(self, origin_poly: Optional[MultivariatePolynomialSingleABC] = None):
        if origin_poly is None:
            origin_poly = self.origin_poly
        # TODO check the validity of the basis (basis of the input poly during init, grids match)
        # TODO helper fcts equality of bases ("grid") __eq__

        elif type(origin_poly) != self.origin_type:
            raise TypeError(
                f"Input polynomial type <{type(origin_poly)}> differs from expected polynomial type <{self.origin_type}>")
        # TODO unsafe. user could input wrong polynomial (e.g. different multi_index!)
        #   or even worse: with same multi index but different grid! (undetected!)
        #   -> test input match!
        return self._apply_transformation(origin_poly)

    @property
    @abstractmethod
    def origin_type(self):
        pass

    @property
    @abstractmethod
    def target_type(self):
        pass

    @property
    @abstractmethod
    def _short_name(self):
        pass

    @abstractmethod
    def _get_transformation_operator(self):
        pass

    @property
    def transformation_operator(self) -> TransformationOperatorABC:
        if self._transformation_operator is None:
            self._transformation_operator: TransformationOperatorABC = self._get_transformation_operator()
        return self._transformation_operator


    @property
    def _target_indices(self) -> MultiIndex:
        """
        :return: the indices the target polynomial will have

        NOTE: poly.multi_index and poly.grid.multi_index might not be equal!
        this is required since e.g. transforming a polynomial in Lagrange basis
            into Newton basis possibly "activates" all indices of the basis (grid).

        TODO more specifically: only all "previous" = lexicographically smaller indices will be active
            -> same as completing only the active multi indices.
        ATTENTION: the multi indices of the basis in use must stay equal!

        """
        return self.origin_poly.grid.multi_index

    def _apply_transformation(self, origin_poly):
        # TODO discuss: is it meaningful to create a new polynomial instance every time?
        #  perhaps allow an optional output polynomial as parameter and then just update the coefficients?
        # NOTE: construct a new polynomial from the input polynomial in order to copy all relevant internal attributes!
        output_poly = self.target_type.from_poly(origin_poly)
        # ATTENTION: assign the correct expected multi indices!
        output_poly.multi_index = self._target_indices
        # NOTE: only then the coefficients can be assigned, since the shapes need to match with the indices!
        # NOTE: this is calling self.transformation_operator.__matmul__(origin_poly.coeffs)
        target_coeffs = self.transformation_operator @ origin_poly.coeffs
        output_poly.coeffs = target_coeffs
        return output_poly
