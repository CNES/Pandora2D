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


def get_invalid_disp_mask(row_map: NDArray, col_map: NDArray, invalid_disp: Union[int, float]) -> NDArray:
    """
    Compute mask for points equal to invalid_disp in row_map or col_map.
    This mask is then used to remove invalid points from position vectors.

    :param row_map: row disparity map
    :type row_map: NDArray
    :param col_map: col disparity map
    :type col_map: NDArray
    :param invalid_disp: invalid disparity value
    :type invalid_disp: Union[int, float]
    :return: invalid mask for position matrix
    :rtype: NDArray
    """

    if np.isnan(invalid_disp):
        mask_invalid = np.isnan(row_map) | np.isnan(col_map)
    elif np.isinf(invalid_disp):
        mask_invalid = np.isinf(row_map) | np.isinf(col_map)
    else:
        mask_invalid = (row_map == invalid_disp) | (col_map == invalid_disp)

    return mask_invalid


def make_position_vectors(dataset_disp_maps: xr.Dataset) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Construct initial et final positions vectors using dataset_disp_maps coordinates
    and disparity maps.

    For the least squares problem y=Xb:
        - X is constructed using init_row_coords and init_col_coords
        - y is either final_row_coords or final_col_coords.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :return: initial position vectors and final position vectors
    :rtype: Tuple[NDArray, NDArray]
    """

    # dataset_disp_maps xarray coordinates are used to get initial positions
    # for each point of the image
    col_coords = dataset_disp_maps["col"].values
    row_coords = dataset_disp_maps["row"].values

    init_col_coords, init_row_coords = np.meshgrid(col_coords, row_coords)

    # final positions are computed using initial positions
    # to which we add the disparities calculated by pandora2d
    final_col_coords = init_col_coords + dataset_disp_maps["col_map"].data
    final_row_coords = init_row_coords + dataset_disp_maps["row_map"].data

    # We remove the points that are invalid_disp either in row_map or in col_map
    # so as not to distort the calculation of the least squares coefficients
    mask_invalid = get_invalid_disp_mask(
        dataset_disp_maps["row_map"].data, dataset_disp_maps["col_map"].data, dataset_disp_maps.attrs["invalid_disp"]
    )

    init_row_coords = init_row_coords[~mask_invalid]
    init_col_coords = init_col_coords[~mask_invalid]
    final_row_coords = final_row_coords[~mask_invalid]
    final_col_coords = final_col_coords[~mask_invalid]

    return init_row_coords, init_col_coords, final_row_coords, final_col_coords


def make_polynomial_design_matrix(
    row_init_coords: NDArray, col_init_coords: NDArray, degree: int
) -> Tuple[NDArray, List]:
    """
    Construct a 2D polynomial design matrix up to a given degree.

    It generates all polynomial combination of the input variables col_init_coords
    and row_init_coords.

    For the least squares problem y=Xb, X is the design matrix.

    :param col_init_coords: initial column positions
    :type col_init_coords: NDArray
    :param row_init_coords: initial row positions
    :type row_init_coords: NDArray
    :param degree: polynomial degree
    :type degree: int
    :return: polynomial design matrix and exponent pairs list
    :rtype: Tuple[NDArray, List]
    """

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
    Estimate deformation model from initial positions to final positions
    by resolving y=Xb.

    :param degree: polynomial degree
    :type degree: int
    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :return: least square solution and sum of residuals for rows and columns
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """

    # Get initial and final position vectors
    row_init_coords, col_init_coords, row_final_coords, col_final_coords = make_position_vectors(dataset_disp_maps)
    # Get design matrix X
    design_matrix, exponent_pairs = make_polynomial_design_matrix(row_init_coords, col_init_coords, degree)

    # Check that we have enough observations compared to the number of parameters
    check_nb_observations(design_matrix)

    # Resolve least squares problem y = Xb
    coefficients_row, sum_sq_residuals_row, _, __ = np.linalg.lstsq(design_matrix, row_final_coords.ravel())
    coefficients_col, sum_sq_residuals_col, _, __ = np.linalg.lstsq(design_matrix, col_final_coords.ravel())

    return coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, exponent_pairs


def estimate_model_cholesky(
    dataset_disp_maps: xr.Dataset, degree: int, lambda_ridge: Union[int, None] = None
) -> Tuple[NDArray, NDArray, NDArray, NDArray, List]:
    """
    Estimate deformation model from initial positions to final positions
    by resolving y=Xb using Cholesky decomposition.

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

    # Get initial and final position vectors
    row_init_coords, col_init_coords, row_final_coords, col_final_coords = make_position_vectors(dataset_disp_maps)
    # Get design matrix X
    design_matrix, exponent_pairs = make_polynomial_design_matrix(row_init_coords, col_init_coords, degree)

    # Check that we have enough observations compared to the number of parameters
    check_nb_observations(design_matrix)

    # Add ridge penalty if the lambda_ridge parameter is specified
    # then compute Cholesky matrix L such as X.T*X = L*L.T
    if lambda_ridge is None:
        cholesky_matrix = np.linalg.cholesky(np.dot(design_matrix.T, design_matrix))
    else:
        ridge_regularisation = lambda_ridge * np.eye(design_matrix.shape[1])
        cholesky_matrix = np.linalg.cholesky(np.dot(design_matrix.T, design_matrix) + ridge_regularisation)

    # Resolve first system L*z=X.T*y
    intermediate_vector_row = np.linalg.solve(cholesky_matrix, np.dot(design_matrix.T, row_final_coords.ravel()))
    intermediate_vector_col = np.linalg.solve(cholesky_matrix, np.dot(design_matrix.T, col_final_coords.ravel()))

    # Resolve second system L.T*b=z
    coefficients_row = np.linalg.solve(cholesky_matrix.T, intermediate_vector_row)
    coefficients_col = np.linalg.solve(cholesky_matrix.T, intermediate_vector_col)

    # Compute sum of squared residuals
    sum_sq_residuals_row = np.sum((row_final_coords.ravel() - np.dot(design_matrix, coefficients_row)) ** 2)
    sum_sq_residuals_col = np.sum((col_final_coords.ravel() - np.dot(design_matrix, coefficients_col)) ** 2)

    return coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, exponent_pairs


def estimate_init_disparity_grids(
    dataset_disp_maps: xr.Dataset,
    coefficients_row: NDArray,
    coefficients_col: NDArray,
    scale_factor: int,  # pylint: disable=unused-argument
    degree: int,
    next_resolution_shape: Tuple,
) -> Tuple[NDArray, NDArray]:
    """
    Estimate initial disparity grids (rows and columns) according to scale factor
    and coefficients of least squares resolution.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param coefficients_row: row coefficients computed by least squares resolution
    :type_coefficients_row: NDArray
    :param coefficients_col: col coefficients computed by least squares resolution
    :type_coefficients_col: NDArray
    :param degree: polynomial degree
    :type degree: int
    :param next_resolution_shape: shape of image for next resolution
    :type next_resolution_shape: Tuple (height, width)
    :return: initial disparity grids for rows and columns
    :rtype: Tuple[NDArray, NDArray]
    """

    # Get resampled coordinates according to scale factor
    scaled_row = np.linspace(
        dataset_disp_maps.coords["row"].values[0],
        dataset_disp_maps.coords["row"].values[-1] + 1,
        next_resolution_shape[0],
        endpoint=False,
    )
    scaled_col = np.linspace(
        dataset_disp_maps.coords["col"].values[0],
        dataset_disp_maps.coords["col"].values[-1] + 1,
        next_resolution_shape[1],
        endpoint=False,
    )

    # Get initial positions for resampled coordinates
    scaled_col_2d, scaled_row_2d = np.meshgrid(scaled_col, scaled_row)
    # Get design matrix for resampled initial positions
    design_matrix, _ = make_polynomial_design_matrix(scaled_row_2d.ravel(), scaled_col_2d.ravel(), degree)

    # Compute the final disparity grids estimated using least squares coefficients
    estimated_final_row = np.dot(design_matrix, coefficients_row)
    estimated_final_col = np.dot(design_matrix, coefficients_col)
    # Reshape estimated final disparity grids
    estimated_final_row_grid = estimated_final_row.reshape(scaled_row_2d.shape)
    estimated_final_col_grid = estimated_final_col.reshape(scaled_col_2d.shape)
    # Compute estimated initial disparity grid for next resolution
    # by subtracting the resampled initial position
    estimated_init_row_grid = np.round(estimated_final_row_grid - scaled_row_2d)
    estimated_init_col_grid = np.round(estimated_final_col_grid - scaled_col_2d)

    return estimated_init_row_grid, estimated_init_col_grid
