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
from pandora2d.constants import Criteria
from pandora2d.multiscale import model_estimation


@pytest.fixture
def invalid_disp():
    return -9999


@pytest.fixture
def dataset_disp_maps(row_coords, col_coords, invalid_disp, data_row_map, data_col_map):
    """
    Disparity maps dataset
    """

    # We add MESH_validity band because methods tested in this file use this band,
    # which is normally added in the `get_init_disparity_grids_with_mesh` method
    criteria_names = (
        ["validity_mask"] + ["partial_validity_mask"] + list(Criteria.__members__.keys())[1:] + ["MESH_validity"]
    )

    coords = {
        "row": row_coords,
        "col": col_coords,
        "criteria": criteria_names,
    }

    dims = ("row", "col")
    dims_validity = ("row", "col", "criteria")
    shape = (len(coords.get("row")), len(coords.get("col")))

    dataset = xr.Dataset(
        {
            "row_map": (dims, data_row_map),
            "col_map": (dims, data_col_map),
            "correlation_score": (dims, np.full(shape, invalid_disp, dtype=np.float32)),
            "validity": (dims_validity, np.full((shape[0], shape[1], len(coords.get("criteria"))), 0, dtype=np.uint8)),
        },
        coords=coords,
    )

    dataset.attrs = {"invalid_disp": invalid_disp, "minimal_nb_pixels_per_mesh": 1}

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
    ["init_row_pos", "init_col_pos", "degree", "design_matrix_gt"],
    [
        pytest.param(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 0, 1, 0, 1]),
            1,
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 2], [1, 1, 2]]),
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
            id="Degree=2",
        ),
    ],
)
def test_make_polynomial_design_matrix(init_row_pos, init_col_pos, degree, design_matrix_gt):
    """
    Test make_polynomial_design_matrix method
    """

    design_matrix = model_estimation.make_polynomial_design_matrix(init_row_pos, init_col_pos, degree)

    np.testing.assert_array_equal(design_matrix, design_matrix_gt)


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

    coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col = method(dataset_disp_maps, degree)

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

    coefficients_row, coefficients_col, sum_sq_residuals_row, sum_sq_residuals_col = (
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
                    [2.0, 3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [7.0, 8.0, 9.0, 10.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [10.0, 11.0, 12.0, 13.0],
                ]
            ),
            np.array(
                [
                    [2.0, 2.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 1.0],
                    [3.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 2.0, 2.0],
                    [3.0, 3.0, 2.0, 2.0],
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
                    [6.0, 7.0, 8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 6.0, 4.0],
                    [7.0, 8.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0],
                    [8.0, 9.0, 10.0, 11.0, 12.0, 12.0, 12.0, 12.0, 11.0, 10.0],
                    [7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 12.0, 12.0],
                    [5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 13.0],
                    [3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0, 12.0, 13.0, 12.0],
                ]
            ),
            np.array(
                [
                    [1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 9.0, 8.0],
                    [1.0, 3.0, 5.0, 6.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0],
                    [2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 10.0, 11.0, 11.0, 11.0],
                    [4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 13.0],
                    [6.0, 9.0, 11.0, 12.0, 14.0, 15.0, 16.0, 16.0, 17.0, 17.0],
                    [10.0, 12.0, 14.0, 16.0, 17.0, 19.0, 20.0, 20.0, 21.0, 21.0],
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

    coefficients_row, coefficients_col, _, __ = model_estimation.estimate_model(dataset_disp_maps, degree)

    estimated_init_row_grid, estimated_init_col_grid = model_estimation.estimate_init_disparity_grids(
        dataset_disp_maps, coefficients_row, coefficients_col, degree, next_resolution_shape
    )

    np.testing.assert_array_equal(estimated_init_row_grid, gt_row_grid)
    np.testing.assert_array_equal(estimated_init_col_grid, gt_col_grid)


@pytest.mark.parametrize(
    ["next_resolution_shape", "nb_mesh", "ground_truth"],
    [
        pytest.param(
            100,
            1,
            [100],
            id="Only one mesh",
        ),
        pytest.param(
            1500,
            2,
            [750, 750],
            id="next_resolution_shape mod nb_mesh = 0",
        ),
        pytest.param(1486, 3, [495, 495, 496], id="next_resolution_shape mod nb_mesh = 1"),
        pytest.param(14, 3, [4, 5, 5], id="next_resolution_shape mod nb_mesh = 2"),
        pytest.param(18, 5, [3, 3, 4, 4, 4], id="next_resolution_shape mod nb_mesh = 3"),
    ],
)
def test_get_next_shape_mesh_list(next_resolution_shape, nb_mesh, ground_truth):
    """
    Test the get_next_shape_mesh_list method
    """

    next_shape_list = model_estimation.get_next_shape_mesh_list(next_resolution_shape, nb_mesh)

    assert next_shape_list == ground_truth


@pytest.mark.parametrize(
    ["estimated_init_grid_list", "nb_row_mesh", "nb_col_mesh", "ground_truth"],
    [
        pytest.param(
            [np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])],
            1,
            1,
            np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
            id="1 mesh in row and 1 mesh in column",
        ),
        pytest.param(
            [
                np.array([[1, 1], [1, 1]]),
                np.array([[2, 2], [2, 2]]),
                np.array([[3, 3], [3, 3]]),
                np.array([[4, 4], [4, 4]]),
            ],
            2,
            2,
            np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
            id="2 mesh in row and 2 mesh in column",
        ),
        pytest.param(
            [
                np.array([[1], [1], [1], [1]]),
                np.array([[2], [2], [2], [2]]),
                np.array([[3, 3], [3, 3], [3, 3], [3, 3]]),
            ],
            1,
            3,
            np.array([[1, 2, 3, 3], [1, 2, 3, 3], [1, 2, 3, 3], [1, 2, 3, 3]]),
            id="1 mesh in row and 3 mesh in column",
        ),
        pytest.param(
            [np.array([[1, 1, 1, 1]]), np.array([[2, 2, 2, 2]]), np.array([[3, 3, 3, 3], [3, 3, 3, 3]])],
            3,
            1,
            np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]]),
            id="3 mesh in row and 1 mesh in column",
        ),
        pytest.param(
            [
                np.array([[1, 1], [1, 1], [1, 1]]),
                np.array([[2, 2], [2, 2], [2, 2]]),
                np.array([[3, 3], [3, 3], [3, 3]]),
                np.array([[4, 4], [4, 4], [4, 4]]),
                np.array([[5, 5], [5, 5], [5, 5]]),
                np.array([[6, 6], [6, 6], [6, 6]]),
            ],
            2,
            3,
            np.array(
                [
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [1, 1, 2, 2, 3, 3],
                    [4, 4, 5, 5, 6, 6],
                    [4, 4, 5, 5, 6, 6],
                    [4, 4, 5, 5, 6, 6],
                ]
            ),
            id="2 mesh in row and 3 mesh in column",
        ),
        pytest.param(
            [
                np.array([[1, 1, 1]]),
                np.array([[2, 2, 2]]),
                np.array([[3, 3, 3]]),
                np.array([[4, 4, 4]]),
                np.array([[5, 5, 5], [5, 5, 5]]),
                np.array([[6, 6, 6], [6, 6, 6]]),
                np.array([[7, 7, 7], [7, 7, 7]]),
                np.array([[8, 8, 8], [8, 8, 8]]),
            ],
            4,
            2,
            np.array(
                [
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                ]
            ),
            id="4 mesh in row and 2 mesh in column",
        ),
    ],
)
def test_concatenate_estimated_grids(estimated_init_grid_list, nb_row_mesh, nb_col_mesh, ground_truth):
    """
    Test the concatenate_estimated_grids method
    """

    estimated_init_grid = model_estimation.concatenate_estimated_grids(
        estimated_init_grid_list, nb_row_mesh, nb_col_mesh
    )

    np.testing.assert_array_equal(estimated_init_grid, ground_truth)
