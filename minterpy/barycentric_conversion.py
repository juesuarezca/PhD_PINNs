#!/usr/bin/env python
""" functions required for converting different transformation formats into each other
    and to construct a full size array for testin purposes (comparison with the regular conversion matrices)

TODO implement all other conversion functions
"""

__author__ = "Jannik Michelfeit"
__copyright__ = "Copyright 2021, minterpy"
__credits__ = ["Jannik Michelfeit"]
# __license__ =
# __version__ =
# __maintainer__ =
__email__ = "jannik@michelfe.it"
__status__ = "Development"

import numpy as np
from numba import njit

from minterpy.global_settings import ARRAY, INT_DTYPE, FLOAT_DTYPE, TRAFO_DICT, TYPED_LIST


@njit(cache=True)
def merge_trafo_dict(trafo_dict: TRAFO_DICT, leaf_positions: ARRAY) -> ARRAY:
    """ creates a transformation array of full size from a precomputed barycentric transformation in dictionary format

    TODO use the same merging fct everywhere in order to remove redundancies
    TODO convert into piecewise format first, create fct for this,
    """
    last_leaf_idx = len(leaf_positions) - 1
    last_leaf_pos = leaf_positions[last_leaf_idx]
    last_leaf_size = trafo_dict[last_leaf_idx, last_leaf_idx].shape[0]
    expected_size = last_leaf_pos + last_leaf_size
    combined_matrix = np.zeros((expected_size, expected_size), dtype=FLOAT_DTYPE)
    for (leaf_idx_l, leaf_idx_r), matrix_piece, in trafo_dict.items():
        start_pos_in = leaf_positions[leaf_idx_l]
        start_pos_out = leaf_positions[leaf_idx_r]

        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation array piece!
        size_out, size_in = matrix_piece.shape
        end_pos_in = start_pos_in + size_in
        end_pos_out = start_pos_out + size_out

        window = combined_matrix[start_pos_out:end_pos_out, start_pos_in:end_pos_in]
        window[:] = matrix_piece

    return combined_matrix


@njit(cache=True)
def factorised_2_piecewise(first_leaf_solution, leaf_factors, leaf_positions, leaf_sizes):
    """ computes the actual matrix pieces of a transformation in factorised format explicitly

    NOTE:  useful e.g. for merging all the pieces into a single matrix
    """
    matrix_pieces = []
    start_positions_1 = []
    start_positions_2 = []
    nr_of_leaves = len(leaf_positions)
    for node_idx_1 in range(nr_of_leaves):
        # "lower triangular form"
        for node_idx_2 in range(node_idx_1, nr_of_leaves):
            corr_factor = leaf_factors[node_idx_2, node_idx_1]
            if corr_factor == 0.0:
                continue

            size_in = leaf_sizes[node_idx_1]
            size_out = leaf_sizes[node_idx_2]

            transformation_piece = first_leaf_solution[:size_out, :size_in] * corr_factor

            matrix_pieces.append(transformation_piece)

            start_pos_in = leaf_positions[node_idx_1]
            start_pos_out = leaf_positions[node_idx_2]
            start_positions_1.append(start_pos_in)
            start_positions_2.append(start_pos_out)

    start_positions_1 = np.array(start_positions_1, dtype=INT_DTYPE)
    start_positions_2 = np.array(start_positions_2, dtype=INT_DTYPE)

    return matrix_pieces, start_positions_1, start_positions_2


@njit(cache=True)
def merge_trafo_piecewise(matrix_pieces: TYPED_LIST, start_positions_in: ARRAY, start_positions_out: ARRAY) -> ARRAY:
    """ creates a transformation array of full size from a precomputed barycentric transformation in piecewise format

    used for testing the equality of the transformation matrices of both regular and barycentric computation
    TODO allow to only create a slice of the total matrix
    """
    # ATTENTION: the last entry must belong to the last leaf node
    # -> the ordering of the matrix pieces is not irrelevant!
    last_leaf_idx = len(matrix_pieces) - 1
    last_leaf_pos = start_positions_in[last_leaf_idx]
    last_leaf_size = matrix_pieces[last_leaf_idx].shape[0]
    expected_size = last_leaf_pos + last_leaf_size
    combined_matrix = np.zeros((expected_size, expected_size), dtype=FLOAT_DTYPE)
    for matrix_piece, start_pos_in, start_pos_out in zip(matrix_pieces, start_positions_in, start_positions_out):
        # NOTE: the size of the required slices of the coefficient vectors
        # are implicitly encoded in the size of each transformation array piece!
        size_out, size_in = matrix_piece.shape
        end_pos_in = start_pos_in + size_in
        end_pos_out = start_pos_out + size_out

        window = combined_matrix[start_pos_out:end_pos_out, start_pos_in:end_pos_in]
        window[:] = matrix_piece

    return combined_matrix


@njit(cache=True)
def merge_trafo_factorised(first_leaf_solution: ARRAY, leaf_factors: ARRAY, leaf_positions: ARRAY,
                           leaf_sizes: ARRAY) -> ARRAY:
    """ creates a transformation array of full size from a precomputed barycentric transformation in factorised format
    """
    trafo_piecewise = factorised_2_piecewise(first_leaf_solution, leaf_factors,
                                             leaf_positions, leaf_sizes)
    return merge_trafo_piecewise(*trafo_piecewise)
