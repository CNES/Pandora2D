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
Test multiscale model estimation methods
"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
import xarray as xr
from pandora2d.multiscale import model_estimation


@pytest.fixture
def invalid_disp():
    return -9999


@pytest.fixture
def dataset_disp_maps(row_coords, col_coords, invalid_disp, data_row_map, data_col_map):
    """
    Disparity maps dataset
    """

    coords = {
        "row": row_coords,
        "col": col_coords,
    }

    dims = ("row", "col")
    shape = (len(coords.get("row")), len(coords.get("col")))

    dataset = xr.Dataset(
        {
            "row_map": (dims, data_row_map),
            "col_map": (dims, data_col_map),
            "correlation_score": (dims, np.full(shape, invalid_disp, dtype=np.float32)),
        },
        coords=coords,
    )

    dataset.attrs = {"invalid_disp": invalid_disp}

    return dataset


@pytest.mark.parametrize(
    ["row_coords", "col_coords", "data_row_map", "data_col_map", "gt_row_pos", "gt_col_pos"],
    [
        pytest.param(
            np.arange(3),
            np.arange(5),
            np.zeros((3, 5)),
            np.zeros((3, 5)),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
            id="Classic case",
        ),
        pytest.param(
            np.arange(3),
            np.arange(5),
            np.array([[0, 2, 4, 3, 1], [7, 1, 3, 9, 2], [1, 6, 5, 5, 8]]),
            np.array([[0, 0, 2, 7, 4], [0, 4, 2, 3, 4], [0, 1, 9, 3, 8]]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
            id="Filled disparity maps",
        ),
        pytest.param(
            np.arange(2, 5),
            np.arange(10, 15),
            np.array([[0, 2, 4, 3, 1], [7, 1, 3, 9, 2], [1, 6, 5, 5, 8]]),
            np.array([[0, 0, 2, 7, 4], [0, 4, 2, 3, 4], [0, 1, 9, 3, 8]]),
            np.array([2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]),
            np.array([10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14]),
            id="ROI disparity maps",
        ),
        pytest.param(
            np.arange(3),
            np.arange(5),
            np.array([[0, 2, -9999, 3, 1], [7, 1, 3, 9, 2], [-9999, 6, 5, 5, 8]]),
            np.array([[0, 0, 2, 7, 4], [0, 4, 2, -9999, 4], [0, 1, 9, 3, 8]]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 1, 3, 4, 0, 1, 2, 4, 1, 2, 3, 4]),
            id="Invalid disparities in disparity maps",
        ),
    ],
)
def test_make_position_vectors(dataset_disp_maps, gt_row_pos, gt_col_pos):
    """
    Test make_position_vectors method
    """

    row_coords_2d, col_coords_2d, final_row_coords, final_col_coords = model_estimation.make_position_vectors(
        dataset_disp_maps
    )

    # We only want to compare valid points.
    mask_invalid = model_estimation.get_invalid_disp_mask(
        dataset_disp_maps["row_map"].data, dataset_disp_maps["col_map"].data, dataset_disp_maps.attrs["invalid_disp"]
    )

    np.testing.assert_array_equal(row_coords_2d, gt_row_pos)
    np.testing.assert_array_equal(col_coords_2d, gt_col_pos)
    np.testing.assert_array_equal(
        final_row_coords, row_coords_2d + dataset_disp_maps["row_map"].data[~mask_invalid].ravel()
    )
    np.testing.assert_array_equal(
        final_col_coords, col_coords_2d + dataset_disp_maps["col_map"].data[~mask_invalid].ravel()
    )


@pytest.mark.parametrize(
    ["init_row_pos", "init_col_pos", "degree", "design_matrix_gt", "exponent_pairs_gt"],
    [
        pytest.param(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 0, 1, 0, 1]),
            1,
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 2], [1, 1, 2]]),
            # The columns of the matrix above correspond
            # to the product of positions in rows and columns with
            # the following exponents:
            # (r⁰,c⁰) (r⁰,c¹) (r¹,c⁰)
            [(0, 0), (0, 1), (1, 0)],
            id="Degree=1",
        ),
        pytest.param(
            np.array([2, 2, 3, 3, 4, 4]),
            np.array([10, 11, 10, 11, 10, 11]),
            2,
            np.array(
                [
                    [1, 10, 100, 2, 20, 4],
                    [1, 11, 121, 2, 22, 4],
                    [1, 10, 100, 3, 30, 9],
                    [1, 11, 121, 3, 33, 9],
                    [1, 10, 100, 4, 40, 16],
                    [1, 11, 121, 4, 44, 16],
                ]
            ),
            # The columns of the matrix above correspond
            # to the product of positions in rows and columns with
            # the following exponents:
            # (r⁰,c⁰) (r⁰,c¹) (r⁰,c²) (r¹,c⁰) (r¹,c¹) (r²,c⁰)
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            id="Degree=2",
        ),
    ],
)
def test_make_polynomial_design_matrix(init_row_pos, init_col_pos, degree, design_matrix_gt, exponent_pairs_gt):
    """
    Test make_polynomial_design_matrix method
    """

    design_matrix, exponent_pairs = model_estimation.make_polynomial_design_matrix(init_row_pos, init_col_pos, degree)

    np.testing.assert_array_equal(design_matrix, design_matrix_gt)
    assert exponent_pairs == exponent_pairs_gt


@pytest.mark.parametrize(
    ["row_coords", "col_coords", "data_row_map", "data_col_map", "degree", "gt_coeff", "gt_resid", "method"],
    [
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            1,
            (np.array([-2.16666667, 1.0, 2.5]), np.array([1.75, 0.66666667, 0.25])),  # (gt_coeff_row, gt_coeff_col)
            (np.array([0.3333333]), np.array([2.41666667])),  # (gt_resid_row, gt_resid_col)
            model_estimation.estimate_model,
            id="Classic case",
        ),
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            1,
            (np.array([-2.16666667, 1.0, 2.5]), np.array([1.75, 0.66666667, 0.25])),  # (gt_coeff_row, gt_coeff_col)
            (np.array([0.3333333]), np.array([2.41666667])),  # (gt_resid_row, gt_resid_col)
            model_estimation.estimate_model_cholesky,
            id="Cholesky case",
        ),
    ],
)
def test_estimate_model(dataset_disp_maps, degree, gt_coeff, gt_resid, method):
    """
    Test estimate_model and estimate_model_cholesky methods
    """

    coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, _ = method(
        dataset_disp_maps, degree
    )

    np.testing.assert_array_almost_equal(coefficients_row, gt_coeff[0], decimal=8)
    np.testing.assert_array_almost_equal(coefficients_col, gt_coeff[1], decimal=8)
    np.testing.assert_array_almost_equal(sum_sq_residuals_row, gt_resid[0])
    np.testing.assert_array_almost_equal(sum_sq_residuals_col, gt_resid[1])


@pytest.mark.parametrize(
    ["row_coords", "col_coords", "data_row_map", "data_col_map", "degree", "gt_coeff", "gt_resid"],
    [
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            1,
            (
                np.array([0.01818182, 0.45454545, 1.67272727]),
                np.array([0.53181818, 1.04545455, 0.34393939]),
            ),  # (gt_coeff_row, gt_coeff_col)
            (np.array([3.517355]), np.array([2.855739201])),  # (gt_resid_row, gt_resid_col)
            id="Cholesky with Ridge case",
        ),
    ],
)
def test_estimate_model_cholesky_with_ridge(dataset_disp_maps, degree, gt_coeff, gt_resid):
    """
    Test estimate_model_cholesky method with ridge regularization
    """

    coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col, _ = (
        model_estimation.estimate_model_cholesky(dataset_disp_maps, degree, lambda_ridge=2)
    )

    np.testing.assert_array_almost_equal(coefficients_row, gt_coeff[0], decimal=8)
    np.testing.assert_array_almost_equal(coefficients_col, gt_coeff[1], decimal=8)
    np.testing.assert_array_almost_equal(sum_sq_residuals_row, gt_resid[0])
    np.testing.assert_array_almost_equal(sum_sq_residuals_col, gt_resid[1])


@pytest.mark.parametrize(
    ["row_coords", "col_coords", "data_row_map", "data_col_map", "degree", "method"],
    [
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            4,
            model_estimation.estimate_model,
            id="Classic case",
        ),
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            4,
            model_estimation.estimate_model_cholesky,
            id="Cholesky case",
        ),
    ],
)
def test_fails_estimate_model(dataset_disp_maps, degree, method):
    """
    Test that estimate_model and estimate_model_cholesky methods fails when we have more parameters
    than observations to resolve the least squares problem.
    """

    with pytest.raises(ValueError) as exc_info:
        method(dataset_disp_maps, degree)
    assert (
        str(exc_info.value) == "To solve the least squares problem, there must be more observations than parameters. "
        "Please reduce the degree of the polynomial or increase the number of observations."
    )


@pytest.mark.parametrize(
    [
        "row_coords",
        "col_coords",
        "data_row_map",
        "data_col_map",
        "degree",
        "scale_factor",
        "gt_row_grid",
        "gt_col_grid",
    ],
    [
        pytest.param(
            np.arange(3),
            np.arange(2, 4),
            np.array([[0, 1], [1, 2], [3, 4]]),
            np.array([[1, 1], [2, 0], [1, 2]]),
            1,
            2,
            np.array(
                [
                    [-0.16666667, 0.33333333, 0.83333333, 1.33333333],
                    [0.58333333, 1.08333333, 1.58333333, 2.08333333],
                    [1.33333333, 1.83333333, 2.33333333, 2.83333333],
                    [2.08333333, 2.58333333, 3.08333333, 3.58333333],
                    [2.83333333, 3.33333333, 3.83333333, 4.33333333],
                    [3.58333333, 4.08333333, 4.58333333, 5.08333333],
                ]
            ),
            np.array(
                [
                    [1.08333333, 0.91666667, 0.75, 0.58333333],
                    [1.20833333, 1.04166667, 0.875, 0.70833333],
                    [1.33333333, 1.16666667, 1.0, 0.83333333],
                    [1.45833333, 1.29166667, 1.125, 0.95833333],
                    [1.58333333, 1.41666667, 1.25, 1.08333333],
                    [1.70833333, 1.54166667, 1.375, 1.20833333],
                ]
            ),
            id="Degree=1",
        ),
        pytest.param(
            np.arange(3),
            np.arange(5),
            np.array([[0, 2, 4, 3, 1], [7, 1, 3, 9, 2], [1, 6, 5, 5, 8]]),
            np.array([[0, 0, 2, 7, 4], [0, 4, 2, 3, 4], [0, 1, 9, 3, 8]]),
            2,
            2,
            np.array(
                [
                    [
                        1.4952381,
                        1.97857143,
                        2.31904762,
                        2.51666667,
                        2.57142857,
                        2.48333333,
                        2.25238095,
                        1.87857143,
                        1.36190476,
                        0.70238095,
                    ],
                    [
                        2.4202381,
                        3.02857143,
                        3.49404762,
                        3.81666667,
                        3.99642857,
                        4.03333333,
                        3.92738095,
                        3.67857143,
                        3.28690476,
                        2.75238095,
                    ],
                    [
                        2.8952381,
                        3.62857143,
                        4.21904762,
                        4.66666667,
                        4.97142857,
                        5.13333333,
                        5.15238095,
                        5.02857143,
                        4.76190476,
                        4.35238095,
                    ],
                    [
                        2.9202381,
                        3.77857143,
                        4.49404762,
                        5.06666667,
                        5.49642857,
                        5.78333333,
                        5.92738095,
                        5.92857143,
                        5.78690476,
                        5.50238095,
                    ],
                    [
                        2.4952381,
                        3.47857143,
                        4.31904762,
                        5.01666667,
                        5.57142857,
                        5.98333333,
                        6.25238095,
                        6.37857143,
                        6.36190476,
                        6.20238095,
                    ],
                    [
                        1.6202381,
                        2.72857143,
                        3.69404762,
                        4.51666667,
                        5.19642857,
                        5.73333333,
                        6.12738095,
                        6.37857143,
                        6.48690476,
                        6.45238095,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        -0.33809524,
                        0.75357143,
                        1.70238095,
                        2.50833333,
                        3.17142857,
                        3.69166667,
                        4.06904762,
                        4.30357143,
                        4.3952381,
                        4.34404762,
                    ],
                    [
                        -0.68809524,
                        0.44107143,
                        1.42738095,
                        2.27083333,
                        2.97142857,
                        3.52916667,
                        3.94404762,
                        4.21607143,
                        4.3452381,
                        4.33154762,
                    ],
                    [
                        -0.63809524,
                        0.52857143,
                        1.55238095,
                        2.43333333,
                        3.17142857,
                        3.76666667,
                        4.21904762,
                        4.52857143,
                        4.6952381,
                        4.71904762,
                    ],
                    [
                        -0.18809524,
                        1.01607143,
                        2.07738095,
                        2.99583333,
                        3.77142857,
                        4.40416667,
                        4.89404762,
                        5.24107143,
                        5.4452381,
                        5.50654762,
                    ],
                    [
                        0.66190476,
                        1.90357143,
                        3.00238095,
                        3.95833333,
                        4.77142857,
                        5.44166667,
                        5.96904762,
                        6.35357143,
                        6.5952381,
                        6.69404762,
                    ],
                    [
                        1.91190476,
                        3.19107143,
                        4.32738095,
                        5.32083333,
                        6.17142857,
                        6.87916667,
                        7.44404762,
                        7.86607143,
                        8.1452381,
                        8.28154762,
                    ],
                ]
            ),
            id="Degree=2",
        ),
    ],
)
def test_estimate_init_disparity_grids(dataset_disp_maps, degree, scale_factor, gt_row_grid, gt_col_grid):
    """
    Test the estimate_init_disparity_grids method
    """

    coefficients_row, coefficients_col, _, __, ___ = model_estimation.estimate_model(dataset_disp_maps, degree)

    estimated_init_row_grid, estimated_init_col_grid = model_estimation.estimate_init_disparity_grids(
        dataset_disp_maps, coefficients_row, coefficients_col, scale_factor, degree
    )

    np.testing.assert_array_almost_equal(estimated_init_row_grid, gt_row_grid)
    np.testing.assert_array_almost_equal(estimated_init_col_grid, gt_col_grid)
