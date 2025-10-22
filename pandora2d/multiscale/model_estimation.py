# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA2D
#
#     https://github.com/CNES/Pandora2D
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains methods associated to the estimation model for the MVP
"""

from typing import Tuple, List, Union

import numpy as np
import xarray as xr

from numpy.typing import NDArray


def make_positions_matrix(dataset_disp_maps: xr.Dataset) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Construct initial et final positions maps using dataset_disp_maps coordinates
    and disparity maps.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :return: initial position grids and final position grids
    :rtype: Tuple[NDArray, NDArray]
    """

    col_coords = dataset_disp_maps["col"].values
    row_coords = dataset_disp_maps["row"].values

    col_coords_2d, row_coords_2d = np.meshgrid(col_coords, row_coords)

    final_col_coords = col_coords_2d + dataset_disp_maps["col_map"].data
    final_row_coords = row_coords_2d + dataset_disp_maps["row_map"].data

    return row_coords_2d, col_coords_2d, final_row_coords, final_col_coords


def make_polynomial_design_matrix(
    row_init_coords: NDArray, col_init_coords: NDArray, degree: int
) -> Tuple[NDArray, List]:
    """
    Construct a 2D polynomial design matrix up to a given degree.

    It generates all polynomial combination of the input variables col_init_coords
    and row_init_coords.

    :param col_init_coords: initial column positions
    :type col_init_coords: NDArray
    :param row_init_coords: initial row positions
    :type row_init_coords: NDArray
    :param degree: polynomial degree
    :type degree: int
    :return: polynomial design matrix and exponent pairs list
    :rtype: Tuple[NDArray, List]
    """

    col_init_coords = col_init_coords.ravel()
    row_init_coords = row_init_coords.ravel()

    exponent_pairs = [(a, b) for a in range(degree + 1) for b in range(degree + 1 - a)]

    design_matrix = np.column_stack([col_init_coords**b * row_init_coords**a for (a, b) in exponent_pairs])
    return design_matrix, exponent_pairs


def check_nb_observations(design_matrix: NDArray):
    """
    Check if we have more parameters than observations, in which case a ValueError is raised.

    :param design_matrix: Design matrix
    :type design matrix: NDArray
    """
    if design_matrix.shape[0] < design_matrix.shape[1]:
        raise ValueError(
            "To solve the least squares problem, there must be more observations than parameters. "
            "Please reduce the degree of the polynomial or increase the number of observations."
        )


def estimate_model(dataset_disp_maps: xr.Dataset, degree: int) -> Tuple[NDArray, NDArray, NDArray, NDArray, List]:
    """
    Estimate deformation model from initial positions to final positions.

    :param degree: polynomial degree
    :type degree: int
    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :return: least square solution and sum of residuals for rows and columns
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """

    row_init_coords, col_init_coords, row_final_coords, col_final_coords = make_positions_matrix(dataset_disp_maps)

    design_matrix, exponent_pairs = make_polynomial_design_matrix(row_init_coords, col_init_coords, degree)

    check_nb_observations(design_matrix)

    coefficients_row, sum_sq_residuals_row, _, __ = np.linalg.lstsq(design_matrix, row_final_coords.ravel())
    coefficients_col, sum_sq_residuals_col, _, __ = np.linalg.lstsq(design_matrix, col_final_coords.ravel())

    return coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, exponent_pairs


def estimate_model_cholesky(
    dataset_disp_maps: xr.Dataset, degree: int, lambda_ridge: Union[int, None] = None
) -> Tuple[NDArray, NDArray, NDArray, NDArray, List]:
    """
    Estimate deformation model from initial positions to final positions
    using Cholesky decomposition.

    If a value is specified for lamba_ridge, Ridge regularization is used.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param degree: polynomial degree
    :type degree: int
    :param lambda_ridge: Ridge regularization factor
    :type lambda_ridge: Union[int,None], None by default
    :return: least square solution and sum of residuals for rows and columns
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, List]
    """

    row_init_coords, col_init_coords, row_final_coords, col_final_coords = make_positions_matrix(dataset_disp_maps)

    design_matrix, exponent_pairs = make_polynomial_design_matrix(row_init_coords, col_init_coords, degree)

    check_nb_observations(design_matrix)

    if lambda_ridge is None:
        cholesky_matrix = np.linalg.cholesky(np.dot(design_matrix.T, design_matrix))
    else:
        ridge_regularisation = lambda_ridge * np.eye(design_matrix.shape[1])
        cholesky_matrix = np.linalg.cholesky(np.dot(design_matrix.T, design_matrix) + ridge_regularisation)

    intermediate_vector_row = np.linalg.solve(cholesky_matrix, np.dot(design_matrix.T, row_final_coords.ravel()))
    intermediate_vector_col = np.linalg.solve(cholesky_matrix, np.dot(design_matrix.T, col_final_coords.ravel()))

    coefficients_row = np.linalg.solve(cholesky_matrix.T, intermediate_vector_row)
    coefficients_col = np.linalg.solve(cholesky_matrix.T, intermediate_vector_col)

    sum_sq_residuals_row = np.sum((row_final_coords.ravel() - np.dot(design_matrix, coefficients_row)) ** 2)
    sum_sq_residuals_col = np.sum((col_final_coords.ravel() - np.dot(design_matrix, coefficients_col)) ** 2)

    return coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, exponent_pairs


def estimate_init_disparity_grids(
    dataset_disp_maps: xr.Dataset, coefficients_row: NDArray, coefficients_col: NDArray, scale_factor: int, degree: int
) -> Tuple[NDArray, NDArray]:
    """
    Estimate initial disparity grids (row and columns) according to scale factor
    and coefficients of least squares resolution.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param coefficients_row: row coefficients computed by least squares resolution
    :type_coefficients_row: NDArray
    :param coefficients_col: col coefficients computed by least squares resolution
    :type_coefficients_col: NDArray
    :param degree: polynomial degree
    :type degree: int
    :return: initial disparity grids for rows and columns
    :rtype: Tuple[NDArray, NDArray]
    """

    scaled_row = np.arange(
        dataset_disp_maps.coords["row"].values[0], dataset_disp_maps.coords["row"].values[-1] + 1, 1 / scale_factor
    )
    scaled_col = np.arange(
        dataset_disp_maps.coords["col"].values[0], dataset_disp_maps.coords["col"].values[-1] + 1, 1 / scale_factor
    )

    scaled_col_2d, scaled_row_2d = np.meshgrid(scaled_col, scaled_row)

    design_matrix, _ = make_polynomial_design_matrix(scaled_row_2d, scaled_col_2d, degree)

    estimated_final_row = np.dot(design_matrix, coefficients_row)
    estimated_final_col = np.dot(design_matrix, coefficients_col)

    estimated_final_row_grid = estimated_final_row.reshape(scaled_row_2d.shape)
    estimated_final_col_grid = estimated_final_col.reshape(scaled_col_2d.shape)

    estimated_init_row_grid = estimated_final_row_grid - scaled_row_2d
    estimated_init_col_grid = estimated_final_col_grid - scaled_col_2d

    return estimated_init_row_grid, estimated_init_col_grid
