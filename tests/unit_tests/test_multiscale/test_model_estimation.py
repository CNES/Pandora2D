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

    np.testing.assert_array_equal(row_coords_2d, gt_row_pos * model_estimation.COMPRESSION_FACTOR)
    np.testing.assert_array_equal(col_coords_2d, gt_col_pos * model_estimation.COMPRESSION_FACTOR)
    np.testing.assert_array_equal(
        final_row_coords,
        row_coords_2d + dataset_disp_maps["row_map"].data[~mask_invalid].ravel() * model_estimation.COMPRESSION_FACTOR,
    )
    np.testing.assert_array_equal(
        final_col_coords,
        col_coords_2d + dataset_disp_maps["col_map"].data[~mask_invalid].ravel() * model_estimation.COMPRESSION_FACTOR,
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
            np.arange(3) / model_estimation.COMPRESSION_FACTOR,
            np.arange(2, 4) / model_estimation.COMPRESSION_FACTOR,
            np.array([[0, 1], [1, 2], [3, 4]]) / model_estimation.COMPRESSION_FACTOR,
            np.array([[1, 1], [2, 0], [1, 2]]) / model_estimation.COMPRESSION_FACTOR,
            1,
            (np.array([-2.16666667, 1.0, 2.5]), np.array([1.75, 0.66666667, 0.25])),  # (gt_coeff_row, gt_coeff_col)
            (np.array([0.3333333]), np.array([2.41666667])),  # (gt_resid_row, gt_resid_col)
            model_estimation.estimate_model,
            id="Classic case",
        ),
        pytest.param(
            np.arange(3) / model_estimation.COMPRESSION_FACTOR,
            np.arange(2, 4) / model_estimation.COMPRESSION_FACTOR,
            np.array([[0, 1], [1, 2], [3, 4]]) / model_estimation.COMPRESSION_FACTOR,
            np.array([[1, 1], [2, 0], [1, 2]]) / model_estimation.COMPRESSION_FACTOR,
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
            np.arange(3) / model_estimation.COMPRESSION_FACTOR,
            np.arange(2, 4) / model_estimation.COMPRESSION_FACTOR,
            np.array([[0, 1], [1, 2], [3, 4]]) / model_estimation.COMPRESSION_FACTOR,
            np.array([[1, 1], [2, 0], [1, 2]]) / model_estimation.COMPRESSION_FACTOR,
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
        "next_resolution_shape",
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
            (6, 4),
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [7.0, 8.0, 9.0, 10.0],
                ]
            ),
            np.array(
                [
                    [2.0, 2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0],
                    [3.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 2.0],
                    [3.0, 3.0, 3.0, 2.0],
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
            (6, 10),
            np.array(
                [
                    [3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 1.0],
                    [5.0, 6.0, 7.0, 8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 6.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0],
                    [6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.0, 12.0, 12.0, 11.0],
                    [5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 12.0],
                    [3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0],
                ]
            ),
            np.array(
                [
                    [-1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 9.0],
                    [-1.0, 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 9.0],
                    [-1.0, 1.0, 3.0, 5.0, 6.0, 8.0, 8.0, 9.0, 9.0, 9.0],
                    [-0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 10.0, 11.0, 11.0],
                    [1.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0],
                    [4.0, 6.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 16.0, 17.0],
                ]
            ),
            id="Degree=2",
        ),
    ],
)
def test_estimate_init_disparity_grids(dataset_disp_maps, degree, next_resolution_shape, gt_row_grid, gt_col_grid):
    """
    Test the estimate_init_disparity_grids method
    """

    coefficients_row, coefficients_col, _, __, ___ = model_estimation.estimate_model(dataset_disp_maps, degree)

    estimated_init_row_grid, estimated_init_col_grid = model_estimation.estimate_init_disparity_grids(
        dataset_disp_maps, coefficients_row, coefficients_col, degree, next_resolution_shape
    )

    np.testing.assert_array_almost_equal(estimated_init_row_grid, gt_row_grid)
    np.testing.assert_array_almost_equal(estimated_init_col_grid, gt_col_grid)
