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

from typing import Tuple, List, Union, Dict

import numpy as np
import xarray as xr

from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

COMPRESSION_FACTOR = 0.5


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
    To reduce numerical instability, a compression factor is used to create the final and initial position grids.

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
    col_coords = dataset_disp_maps["col"].values * COMPRESSION_FACTOR
    row_coords = dataset_disp_maps["row"].values * COMPRESSION_FACTOR

    init_col_coords, init_row_coords = np.meshgrid(col_coords, row_coords)

    # final positions are computed using initial positions
    # to which we add the disparities calculated by pandora2d
    final_col_coords = init_col_coords + dataset_disp_maps["col_map"].data * COMPRESSION_FACTOR
    final_row_coords = init_row_coords + dataset_disp_maps["row_map"].data * COMPRESSION_FACTOR

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
    dataset_disp_maps: xr.Dataset, degree: int, lambda_ridge: Union[int, float, None] = None
) -> Tuple[NDArray, NDArray, NDArray, NDArray, List]:
    """
    Estimate deformation model from initial positions to final positions
    by resolving y=Xb using Cholesky decomposition.

    We have X.T*X*b = X.T*y and X.T*X = L*L.T with L a lower triangular matrix.
    Then, using Cholesky decomposition we can resolve L*z=X.T*y and L.T*b = z.

    If a value is specified for lamba_ridge, Ridge regularization is used.

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param degree: polynomial degree
    :type degree: int
    :param lambda_ridge: Ridge regularization factor
    :type lambda_ridge: Union[int, float, None], None by default
    :return: least square solution and sum of residuals for rows and columns
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, List]
    """

    # Get initial and final position vectors
    row_init_coords, col_init_coords, row_final_coords, col_final_coords = make_position_vectors(dataset_disp_maps)
    # Get design matrix X
    design_matrix, exponent_pairs = make_polynomial_design_matrix(row_init_coords, col_init_coords, degree)

    # Check that we have enough observations compared to the number of parameters
    check_nb_observations(design_matrix)

    # Build the normal matrix (X.T*X) and right-hand sides (X.T*y)
    x_transpose_x = np.dot(design_matrix.T, design_matrix).astype(float)
    x_transpose_y_row = np.dot(design_matrix.T, row_final_coords.ravel())
    x_transpose_y_col = np.dot(design_matrix.T, col_final_coords.ravel())

    # Add ridge penalty if the lambda_ridge parameter is specified
    # then compute Cholesky matrix L such as X.T*X = L*L.T
    if lambda_ridge is not None:
        x_transpose_x += lambda_ridge * np.eye(design_matrix.shape[1])

    # Compute the Cholesky factorization (L is lower triangular)
    c, low = cho_factor(x_transpose_x, lower=True)

    # Resolve Cholesky system
    coefficients_row = cho_solve((c, low), x_transpose_y_row)
    coefficients_col = cho_solve((c, low), x_transpose_y_col)

    # Compute sum of squared residuals
    sum_sq_residuals_row = np.sum((row_final_coords.ravel() - np.dot(design_matrix, coefficients_row)) ** 2)
    sum_sq_residuals_col = np.sum((col_final_coords.ravel() - np.dot(design_matrix, coefficients_col)) ** 2)

    return coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, exponent_pairs


def estimate_init_disparity_grids(
    dataset_disp_maps: xr.Dataset,
    coefficients_row: NDArray,
    coefficients_col: NDArray,
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
    # To reduce numerical instability, a compression factor is used to create resampled coordinates
    scaled_row = (
        np.linspace(
            dataset_disp_maps.coords["row"].values[0],
            dataset_disp_maps.coords["row"].values[-1] + 1,
            next_resolution_shape[0],
            endpoint=False,
        )
        * COMPRESSION_FACTOR
    )
    scaled_col = (
        np.linspace(
            dataset_disp_maps.coords["col"].values[0],
            dataset_disp_maps.coords["col"].values[-1] + 1,
            next_resolution_shape[1],
            endpoint=False,
        )
        * COMPRESSION_FACTOR
    )

    # We add 0.5 to scaled_col and scaled_row because we want to consider the center of the pixels
    # and not their upper left corner.
    scaled_row += 0.5 * COMPRESSION_FACTOR
    scaled_col += 0.5 * COMPRESSION_FACTOR

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
    # Compute estimated initial disparity grid for next resolution by subtracting the resampled initial position,
    # multiplying by the scale factor and dividing by the compression factor
    scale_factor_row = np.round(next_resolution_shape[0] / len(dataset_disp_maps.coords["row"].values))
    scale_factor_col = np.round(next_resolution_shape[1] / len(dataset_disp_maps.coords["col"].values))
    estimated_init_row_grid = np.round(
        (estimated_final_row_grid - scaled_row_2d) * scale_factor_row / COMPRESSION_FACTOR
    )
    estimated_init_col_grid = np.round(
        (estimated_final_col_grid - scaled_col_2d) * scale_factor_col / COMPRESSION_FACTOR
    )

    return estimated_init_row_grid, estimated_init_col_grid


def get_next_shape_mesh_list(next_resolution_shape: int, nb_mesh: int) -> List[int]:
    """
    Return list of shape for mesh for next resolution.
    This list is then used to estimate initial disparity grid at next resolution
    with the right shape for each mesh.

    :param next_resolution_shape: next resolution shape (row or column)
    :type next_resolution_shape: int
    :param nb_mesh: number of mesh (row or column)
    :type nb_mesh: int
    :return: List of shape by mesh (row or column)
    :rtype: List[int]
    """

    next_shape_list = [
        next_resolution_shape // nb_mesh + (1 if mesh >= nb_mesh - (next_resolution_shape % nb_mesh) else 0)
        for mesh in range(nb_mesh)
    ]

    return next_shape_list


def concatenate_estimated_grids(estimated_init_grid_list: List[NDArray], nb_row_mesh: int, nb_col_mesh: int) -> NDArray:
    """
    Concatenate the estimated disparity grids computed by mesh
    and return the full initial disparity grid for next resolution.

    :param estimated_init_grid_list:
    :type estimated_init_grid_list: List[NDArray]
    :param nb_row_mesh: number of mesh for rows
    :type nb_row_mesh: int
    :param nb_col_mesh: number of mesh for columns
    :type nb_col_mesh: int
    :return: full estimated initial disparity grid for next resolution
    :rtype: NDArray
    """

    estimated_init_grid = np.concatenate(
        [
            np.concatenate(estimated_init_grid_list[mesh_row * nb_col_mesh : (mesh_row + 1) * nb_col_mesh], axis=1)
            for mesh_row in range(nb_row_mesh)
        ],
        axis=0,
    )

    return estimated_init_grid


def get_init_disparity_grids(
    dataset_disp_maps: xr.Dataset, multiscale_cfg: Dict, next_resolution_shape: Tuple
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return initial disparity grid after computing least square coefficients

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param multiscale_cfg: multiscale pipeline configuration
    :type multiscale_cfg: Dict
    :param next_resolution_shape: shape of image for next resolution
    :type next_resolution_shape: Tuple (height, width)
    :return: initial disparity grids for rows and columns and sum of squared residuals
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """

    # Estimate model
    coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, _ = estimate_model_cholesky(
        dataset_disp_maps, multiscale_cfg["model"]["degree"], lambda_ridge=1.0e-6
    )

    # Estimate initial disparity grids for next resolution
    estimated_init_row_grid, estimated_init_col_grid = estimate_init_disparity_grids(
        dataset_disp_maps,
        coefficients_row,
        coefficients_col,
        multiscale_cfg["model"]["degree"],
        next_resolution_shape,
    )

    return estimated_init_row_grid, estimated_init_col_grid, sum_sq_residuals_row, sum_sq_residuals_col


def get_init_disparity_grids_with_mesh(
    dataset_disp_maps: xr.Dataset, multiscale_cfg: Dict, next_resolution_shape: Tuple
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return initial disparity grid after computing least square coefficients
    for each mesh

    :param dataset_disp_maps: disparity maps dataset
    :type dataset_disp_maps: xr.Dataset
    :param multiscale_cfg: multiscale pipeline configuration
    :type multiscale_cfg: Dict
    :param next_resolution_shape: shape of image for next resolution
    :type next_resolution_shape: Tuple (height, width)
    :return: initial disparity grids for rows and columns and RMSE for row and columns
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """

    nb_row_mesh = multiscale_cfg["mesh"]["row"]
    nb_col_mesh = multiscale_cfg["mesh"]["col"]

    rows, cols = dataset_disp_maps["row_map"].data.shape

    rows_in_mesh = rows // nb_row_mesh
    cols_in_mesh = cols // nb_col_mesh

    estimated_init_row_grid_list = []
    estimated_init_col_grid_list = []

    # Get shape for next resolution for each mesh
    rows_next_shape_list = get_next_shape_mesh_list(next_resolution_shape[0], nb_row_mesh)
    cols_next_shape_list = get_next_shape_mesh_list(next_resolution_shape[1], nb_col_mesh)

    sum_sq_residuals_row_total = 0.0
    sum_sq_residuals_col_total = 0.0

    for mesh_row in range(nb_row_mesh):
        for mesh_col in range(nb_col_mesh):

            row_start = mesh_row * rows_in_mesh
            col_start = mesh_col * cols_in_mesh

            row_end = rows if mesh_row == nb_row_mesh - 1 else (mesh_row + 1) * rows_in_mesh
            col_end = cols if mesh_col == nb_col_mesh - 1 else (mesh_col + 1) * cols_in_mesh

            # Select sub dataset corresponding to each mesh
            sub_dataset = dataset_disp_maps.isel(row=slice(row_start, row_end), col=slice(col_start, col_end))

            # Compute initial disparity grid by mesh
            sub_estimated_init_row_grid, sub_estimated_init_col_grid, sum_sq_residuals_row, sum_sq_residuals_col = (
                get_init_disparity_grids(
                    sub_dataset, multiscale_cfg, (rows_next_shape_list[mesh_row], cols_next_shape_list[mesh_col])
                )
            )

            estimated_init_row_grid_list.append(sub_estimated_init_row_grid)
            estimated_init_col_grid_list.append(sub_estimated_init_col_grid)

            # Cast in float to avoir mypy error
            sum_sq_residuals_row_total += float(sum_sq_residuals_row)
            sum_sq_residuals_col_total += float(sum_sq_residuals_col)

    # Concatenate mesh to get full initial disparity grids
    estimated_init_row_grid = concatenate_estimated_grids(estimated_init_row_grid_list, nb_row_mesh, nb_col_mesh)
    estimated_init_col_grid = concatenate_estimated_grids(estimated_init_col_grid_list, nb_row_mesh, nb_col_mesh)

    # Compute RMSE
    rmse_row = np.sqrt(sum_sq_residuals_row_total / (dataset_disp_maps["row_map"].size))
    rmse_col = np.sqrt(sum_sq_residuals_col_total / (dataset_disp_maps["row_map"].size))

    return estimated_init_row_grid, estimated_init_col_grid, rmse_row, rmse_col
