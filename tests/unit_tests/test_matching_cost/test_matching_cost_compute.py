# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test compute_cost_volumes method from Matching cost
"""
import importlib.util

# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
import numpy as np
import xarray as xr
from rasterio import Affine

import pytest
from pandora.margins import Margins
from pandora2d import matching_cost
from pandora2d.img_tools import create_datasets_from_inputs, add_disparity_grid


@pytest.mark.parametrize(
    "data_fixture_name",
    [
        "data_with_null_disparity",
        "data_with_positive_disparity_in_col",
        "data_with_positive_disparity_in_row",
        "data_with_negative_disparity_in_col",
        "data_with_negative_disparity_in_row",
        "data_with_disparity_negative_in_row_and_positive_in_col",
    ],
)
@pytest.mark.parametrize("col_step", [1, 2, pytest.param(5, id="Step gt image")])
@pytest.mark.parametrize("row_step", [1, 2, pytest.param(5, id="Step gt image")])
def test_steps(request, data_fixture_name, col_step, row_step):
    """We expect step to work."""
    data = request.getfixturevalue(data_fixture_name)

    # sum of squared difference images self.left, self.right, window_size=3
    cfg = {
        "pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3, "step": [row_step, col_step]}}
    }
    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=data.left,
        img_right=data.right,
        grid_min_col=data.disparity_grids.col_min,
        grid_max_col=data.disparity_grids.col_max,
        cfg=cfg,
    )
    # compute cost volumes
    zncc = matching_cost_matcher.compute_cost_volumes(
        img_left=data.left,
        img_right=data.right,
        grid_min_col=data.disparity_grids.col_min,
        grid_max_col=data.disparity_grids.col_max,
        grid_min_row=data.disparity_grids.row_min,
        grid_max_row=data.disparity_grids.row_max,
    )

    # indexes are : row, col, disp_x, disp_y
    np.testing.assert_equal(zncc["cost_volumes"].data, data.full_matching_cost[::row_step, ::col_step, :, :])


def test_compute_cv_ssd(left_stereo_object, right_stereo_object):
    """
    Test the  cost volume product by ssd
    """
    # sum of squared difference images left, right, window_size=1
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "ssd", "window_size": 1}}}
    # sum of squared difference ground truth for the images left, right, window_size=1
    ad_ground_truth = np.zeros((3, 3, 2, 2))
    # disp_x = -1, disp_y = -1
    ad_ground_truth[:, :, 0, 0] = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, 0, 0], [np.nan, (4 - 3) ** 2, (5 - 4) ** 2]]
    )

    # disp_x = -1, disp_y = 0
    ad_ground_truth[:, :, 0, 1] = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, (1 - 3) ** 2, (1 - 4) ** 2], [np.nan, (4 - 1) ** 2, (5 - 1) ** 2]]
    )

    # disp_x = 0, disp_y = 0
    ad_ground_truth[:, :, 1, 1] = np.array(
        [
            [np.nan, np.nan, np.nan],
            [(1 - 3) ** 2, (1 - 4) ** 2, (1 - 5) ** 2],
            [(3 - 1) ** 2, (4 - 1) ** 2, (5 - 1) ** 2],
        ]
    )

    # disp_x = 0, disp_y = -1
    ad_ground_truth[:, :, 1, 0] = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [0, 0, 0]])
    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        cfg=cfg,
    )

    # compute cost volumes
    ssd = matching_cost_matcher.compute_cost_volumes(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )

    # check that the generated cost_volumes is equal to ground truth
    np.testing.assert_allclose(ssd["cost_volumes"].data, ad_ground_truth, atol=1e-06)


@pytest.mark.usefixtures("import_plugins")
@pytest.mark.plugin_tests
@pytest.mark.skipif(importlib.util.find_spec("mc_cnn") is None, reason="MCCNN plugin not installed")
def test_compute_cv_mc_cnn():
    """
    Test the  cost volume product by mccnn
    """

    # Applied MCCNN on same data, window_size=11
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "mc_cnn", "window_size": 11}}}

    data = np.arange(10000).reshape(100, 100)

    img = xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    img.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "col_disparity_source": [-1, 1],
        "row_disparity_source": [-1, 1],
    }

    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img,
        img_right=img,
        grid_min_col=np.full((100, 100), -1),
        grid_max_col=np.full((100, 100), 1),
        cfg=cfg,
    )

    # compute cost volumes
    mccnn = matching_cost_matcher.compute_cost_volumes(
        img_left=img,
        img_right=img,
        grid_min_col=np.full((100, 100), -1),
        grid_max_col=np.full((100, 100), 1),
        grid_min_row=np.full((100, 100), -1),
        grid_max_row=np.full((100, 100), 1),
    )

    # get cv with disparity = 0
    disp = abs(mccnn["cost_volumes"].data[:, :, 1, 1])
    # check that the correlation score is close to 1
    error_mean = np.nanmean(abs(disp - 1))

    np.testing.assert_allclose(error_mean, 0, atol=1e-06)


def test_compute_cv_sad(left_stereo_object, right_stereo_object):
    """
    Test the  cost volume product by sad
    """

    # sum of squared difference images left, right, window_size=1
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "sad", "window_size": 1}}}
    # sum of absolute difference ground truth for the images left, right, window_size=1
    ad_ground_truth = np.zeros((3, 3, 2, 2))
    # disp_x = -1, disp_y = -1
    ad_ground_truth[:, :, 0, 0] = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, 0, 0], [np.nan, np.abs(4 - 3), np.abs(5 - 4)]]
    )

    # disp_x = -1, disp_y = 0
    ad_ground_truth[:, :, 0, 1] = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.abs(1 - 3), np.abs(1 - 4)], [np.nan, np.abs(4 - 1), np.abs(5 - 1)]]
    )

    # disp_x = 0, disp_y = 0
    ad_ground_truth[:, :, 1, 1] = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.abs(1 - 3), np.abs(1 - 4), np.abs(1 - 5)],
            [np.abs(3 - 1), np.abs(4 - 1), np.abs(5 - 1)],
        ]
    )

    # disp_x = 0, disp_y = -1
    ad_ground_truth[:, :, 1, 0] = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [0, 0, 0]])

    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        cfg=cfg,
    )
    # compute cost volumes
    sad = matching_cost_matcher.compute_cost_volumes(
        img_left=left_stereo_object,
        img_right=right_stereo_object,
        grid_min_col=np.full((3, 3), -1),
        grid_max_col=np.full((3, 3), 0),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )
    # check that the generated cost_volumes is equal to ground truth
    np.testing.assert_allclose(sad["cost_volumes"].data, ad_ground_truth, atol=1e-06)


def test_compute_cv_zncc():
    """
    Test the  cost volume product by zncc
    """
    data = np.array(
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    left_zncc = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left_zncc.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "col_disparity_source": [0, 1],
        "row_disparity_source": [-1, 0],
    }

    data = np.array(
        ([[1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    right_zncc = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right_zncc.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    # sum of squared difference images left, right, window_size=3
    cfg = {"pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3}}}
    # sum of absolute difference ground truth for the images left, right, window_size=1

    left = left_zncc["im"].data
    right = right_zncc["im"].data
    right_shift = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, 1, 1, 1, 1],
            [3, 4, 5, 6, 7],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    # row = 1, col = 1, disp_x = 0, disp_y = 0, ground truth equal -0,45
    ad_ground_truth_1_1_0_0 = (
        np.mean(left[0:3, 0:3] * right[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right[0:3, 0:3]))
    ) / (np.std(left[0:3, 0:3]) * np.std(right[0:3, 0:3]))
    # row = 1, col = 1, disp_x = 0, disp_y = -1, , ground truth equal NaN
    ad_ground_truth_1_1_0_1 = (
        np.mean(left[0:3, 0:3] * right_shift[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right_shift[0:3, 0:3]))
    ) / (np.std(left[0:3, 0:3]) * np.std(right_shift[0:3, 0:3]))
    # row = 2, col = 2, disp_x = 0, disp_y = 0, , ground truth equal -0,47
    ad_ground_truth_2_2_0_0 = (
        np.mean(left[1:4, 1:4] * right[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right[1:4, 1:4]))
    ) / (np.std(left[1:4, 1:4]) * np.std(right[1:4, 1:4]))
    # row = 2, col = 2, disp_x = 0, disp_y = -1, ground truth equal 1
    ad_ground_truth_2_2_0_1 = (
        np.mean(left[1:4, 1:4] * right_shift[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right_shift[1:4, 1:4]))
    ) / (np.std(left[1:4, 1:4]) * np.std(right_shift[1:4, 1:4]))

    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=left_zncc,
        img_right=right_zncc,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        cfg=cfg,
    )
    # compute cost volumes
    zncc = matching_cost_matcher.compute_cost_volumes(
        img_left=left_zncc,
        img_right=right_zncc,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )
    # check that the generated cost_volumes is equal to ground truth

    np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 1], ad_ground_truth_1_1_0_0, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 0], ad_ground_truth_1_1_0_1, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 1], ad_ground_truth_2_2_0_0, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 0], ad_ground_truth_2_2_0_1, rtol=1e-06)


@pytest.mark.parametrize(
    ["roi", "step", "col_expected", "row_expected"],
    [
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [1, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.arange(1, 8),  # Coordinates of user ROI + margins
            id="ROI and step=[1,1]",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 5, "last": 6}, "margins": [2, 3, 2, 3]},
            [2, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.arange(3, 10, 2),  # ROI["row"]["first"]=5 then coordinates are [3, 5, 7, 9]
            id="ROI and step_row < up margin",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 3, 2, 3]},
            [3, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.arange(0, 9, 3),  # ROI["row"]["first"]=3 then coordinates are [0, 3, 6]
            id="ROI and step_row = up margin",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 4, "last": 5}, "margins": [2, 4, 2, 4]},
            [2, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.arange(0, 10, 2),  # ROI["row"]["first"]=4 then coordinates are [0, 2, 4, 6, 8]
            id="ROI and up margin % step_row = 0",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [3, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.arange(3, 8, 3),  # ROI["row"]["first"]=3 then coordinates are [3, 6]
            id="ROI and step_row > up margin",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [9, 1],
            np.arange(1, 8),  # Coordinates of user ROI + margins
            np.array([3]),  # Only ROI["row"]["first"]=3 is in the cost_volume rows
            id="ROI and step_row greater than the number of rows in the ROI",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [3, 2, 3, 2]},
            [1, 2],
            np.arange(1, 9, 2),  # ROI["col"]["first"]=3 then coordinates are [1, 3, 5, 7]
            np.arange(1, 8),  # Coordinates of user ROI + margins
            id="ROI and step_col < left margin",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [1, 2],
            np.arange(1, 8, 2),  # ROI["col"]["first"]=3 then coordinates are [1, 3, 5, 7]
            np.arange(1, 8),  # Coordinates of user ROI + margins
            id="ROI and step_col = left margin",
        ),
        pytest.param(
            {"col": {"first": 4, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [4, 2, 4, 2]},
            [1, 2],
            np.arange(0, 10, 2),  # ROI["col"]["first"]=4 then coordinates are [0, 2, 4, 6, 8]
            np.arange(1, 8),  # Coordinates of user ROI + margins
            id="ROI and left margin % step_col = 0",
        ),
        pytest.param(
            {"col": {"first": 4, "last": 6}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [1, 3],
            np.arange(4, 9, 3),  # ROI["col"]["first"]=4 then coordinates are [4, 7]
            np.arange(1, 8),  # Coordinates of user ROI + margins
            id="ROI and step_col > left margin",
        ),
        pytest.param(
            {"col": {"first": 4, "last": 6}, "row": {"first": 3, "last": 5}, "margins": [3, 3, 3, 3]},
            [1, 12],
            np.array([4]),  # Only ROI["col"]["first"]=4 is in the cost_volume columns
            np.arange(0, 9),  # Coordinates of user ROI + margins
            id="ROI and step_row greater than the number of columns in the ROI",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 4, "last": 6}, "margins": [3, 2, 3, 2]},
            [3, 2],
            np.arange(1, 9, 2),  # ROI["col"]["first"]=3 then coordinates are [1, 3, 5, 7]
            np.arange(4, 9, 3),  # ROI["row"]["first"]=4 then coordinates are [4, 7]
            id="ROI, step_row=3 and step_col=2",
        ),
        pytest.param(
            {"col": {"first": 2, "last": 5}, "row": {"first": 5, "last": 7}, "margins": [2, 2, 2, 2]},
            [10, 11],
            np.array([2]),  # Only ROI["col"]["first"]=2 is in the cost_volume columns
            np.array([5]),  # Only ROI["row"]["first"]=5 is in the cost_volume rows
            id="ROI and step_row and step_col greater than the number of columns and rows in the ROI",
        ),
    ],
)
def test_cost_volume_coordinates_with_roi(roi, input_config, matching_cost_config, col_expected, row_expected):
    """
    Test that we have the correct cost_volumes coordinates with a ROI
    """

    cfg = {"input": input_config, "pipeline": {"matching_cost": matching_cost_config}, "ROI": roi}

    img_left, img_right = create_datasets_from_inputs(input_config, roi=roi)

    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((10, 10), 0),
        grid_max_col=np.full((10, 10), 1),
        cfg=cfg,
    )

    np.testing.assert_array_equal(matching_cost_matcher.grid_.attrs["col_to_compute"], col_expected)

    # compute cost volumes with roi
    cost_volumes_with_roi = matching_cost_matcher.compute_cost_volumes(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((10, 10), 0),
        grid_max_col=np.full((10, 10), 1),
        grid_min_row=np.full((10, 10), -1),
        grid_max_row=np.full((10, 10), 0),
    )

    np.testing.assert_array_equal(cost_volumes_with_roi["cost_volumes"].coords["col"], col_expected)
    np.testing.assert_array_equal(cost_volumes_with_roi["cost_volumes"].coords["row"], row_expected)


@pytest.mark.parametrize(
    ["step", "col_expected", "row_expected"],
    [
        pytest.param(
            [1, 1],
            np.arange(10),  # Same col coordinates as left image
            np.arange(10),  # Same row coordinates as left image
            id="No ROI, step_row=1 and step_col=1",
        ),
        pytest.param(
            [2, 1],
            np.arange(10),
            np.arange(0, 10, 2),  #   1 < step_row < len(cost_volume["cost_volumes"].coords["row"])
            id="No ROI, step_row=2 and step_col=1",
        ),
        pytest.param(
            [11, 1],
            np.arange(10),
            np.array([0]),  # step_row > len(cost_volume["cost_volumes"].coords["row"]) --> only 1 row
            id="No ROI and step_row greater than the number of rows in the cost volume",
        ),
        pytest.param(
            [1, 3],
            np.arange(0, 10, 3),  #   1 < step_col < len(cost_volume["cost_volumes"].coords["col"])
            np.arange(10),
            id="No ROI, step_row=1 and step_col=3",
        ),
        pytest.param(
            [1, 12],
            np.array([0]),  # step_col > len(cost_volume["cost_volumes"].coords["col"]) --> only 1 col
            np.arange(10),
            id="No ROI and step_col greater than the number of columns in the cost volume",
        ),
        pytest.param(
            [3, 2],
            np.arange(0, 10, 2),
            np.arange(0, 10, 3),
            id="No ROI, step_row=3 and step_col=2",
        ),
        pytest.param(
            [12, 13],
            np.array([0]),  # step_col > len(cost_volume["cost_volumes"].coords["col"]) --> only 1 col
            np.array([0]),  # step_row > len(cost_volume["cost_volumes"].coords["row"]) --> only 1 row
            id="No ROI and step_col and step_row greater than the number of rows and columns in the cost volume",
        ),
    ],
)
def test_cost_volume_coordinates_without_roi(input_config, matching_cost_config, col_expected, row_expected):
    """
    Test that we have the correct cost_volumes coordinates without a ROI
    """

    cfg = {
        "input": input_config,
        "pipeline": {"matching_cost": matching_cost_config},
    }

    img_left, img_right = create_datasets_from_inputs(input_config)

    matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((10, 10), 0),
        grid_max_col=np.full((10, 10), 1),
        cfg=cfg,
    )

    np.testing.assert_array_equal(matching_cost_matcher.grid_.attrs["col_to_compute"], col_expected)

    # compute cost volumes without roi
    cost_volumes = matching_cost_matcher.compute_cost_volumes(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((10, 10), 0),
        grid_max_col=np.full((10, 10), 1),
        grid_min_row=np.full((10, 10), -1),
        grid_max_row=np.full((10, 10), 0),
    )

    np.testing.assert_array_equal(cost_volumes["cost_volumes"].coords["col"], col_expected)
    np.testing.assert_array_equal(cost_volumes["cost_volumes"].coords["row"], row_expected)


@pytest.mark.parametrize(
    [
        "step",
        "expected_shape",
        "expected_shape_roi",
        "cost_volumes_slice",
        "cost_volumes_with_roi_slice",
        "squared_image_size",
    ],
    [
        pytest.param(
            [1, 1],
            (5, 5, 2, 2),
            (5, 4, 2, 2),
            np.s_[2:4, 2:4, :, :],
            np.s_[2:4, 1:3, :, :],
            (5, 5),
            id="Step=[1, 1]",
        ),
        pytest.param(
            [1, 2],
            (5, 3, 2, 2),
            (5, 2, 2, 2),
            np.s_[2:4, 2:4:2, :, :],
            np.s_[2:4, 1:3:2, :, :],
            (5, 5),
            id="Step=[1, 2]",
        ),
        pytest.param(
            [2, 1],
            (3, 5, 2, 2),
            (3, 4, 2, 2),
            np.s_[2:4, 2:4, :, :],
            np.s_[2:4, 1:3, :, :],
            (5, 5),
            id="Step=[2, 1]",
        ),
    ],
)
def test_roi_inside_and_margins_inside(  # pylint: disable=too-many-arguments
    input_config,
    configuration_roi,
    matching_cost_matcher,
    cost_volumes,
    roi,
    expected_shape,
    expected_shape_roi,
    cost_volumes_slice,
    cost_volumes_with_roi_slice,
):
    """
    Test the pandora2d matching cost with roi inside the image
    """

    # crop image with roi
    img_left, img_right = create_datasets_from_inputs(input_config, roi=roi)

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((5, 5), 0),
        grid_max_col=np.full((5, 5), 1),
        cfg=configuration_roi,
    )
    # compute cost volumes with roi
    cost_volumes_with_roi = matching_cost_matcher.compute_cost_volumes(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((5, 5), 0),
        grid_max_col=np.full((5, 5), 1),
        grid_min_row=np.full((5, 5), -1),
        grid_max_row=np.full((5, 5), 0),
    )

    assert cost_volumes_with_roi["cost_volumes"].data.shape == expected_shape_roi
    assert cost_volumes["cost_volumes"].data.shape == expected_shape
    np.testing.assert_array_equal(
        cost_volumes["cost_volumes"].data[cost_volumes_slice],
        cost_volumes_with_roi["cost_volumes"].data[cost_volumes_with_roi_slice],
    )


class TestSubpix:
    """Test subpix parameter"""

    @pytest.fixture()
    def make_image_fixture(self):
        """
        Create image dataset
        """

        def make_image(disp_row, disp_col, data):
            img = xr.Dataset(
                {"im": (["row", "col"], data)},
                coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            )

            add_disparity_grid(img, disp_col, disp_row)

            img.attrs = {
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "col_disparity_source": disp_col,
                "row_disparity_source": disp_row,
            }

            return img

        return make_image

    @pytest.fixture()
    def make_cost_volumes(self, make_image_fixture, request):
        """
        Instantiate a matching_cost and compute cost_volumes
        """

        cfg = {
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "ssd",
                    "window_size": 1,
                    "step": request.param["step"],
                    "subpix": request.param["subpix"],
                }
            }
        }

        disp_row = request.param["disp_row"]
        disp_col = request.param["disp_col"]

        img_left = make_image_fixture(disp_row, disp_col, request.param["data_left"])
        img_right = make_image_fixture(disp_row, disp_col, request.param["data_right"])

        matching_cost_ = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

        matching_cost_.allocate_cost_volume_pandora(
            img_left=img_left,
            img_right=img_right,
            grid_min_col=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_col[0]),
            grid_max_col=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_col[1]),
            cfg=cfg,
        )

        cost_volumes = matching_cost_.compute_cost_volumes(
            img_left=img_left,
            img_right=img_right,
            grid_min_col=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_col[0]),
            grid_max_col=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_col[1]),
            grid_min_row=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_row[0]),
            grid_max_row=np.full((img_left["im"].shape[0], img_left["im"].shape[1]), disp_row[1]),
        )

        return cost_volumes

    @pytest.mark.parametrize(
        ["make_cost_volumes", "shape_expected", "row_disparity", "col_disparity"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 1,
                    "disp_row": [0, 3],
                    "disp_col": [-2, 2],
                    "data_left": np.full((10, 10), 1),
                    "data_right": np.full((10, 10), 1),
                },
                (10, 10, 5, 4),  # (row, col, disp_col, disp_row)
                np.arange(4),  # [0, 1, 2, 3]
                np.arange(-2, 3),  # [-2, -1, 0, 1, 2]
                id="subpix=1, step_row=1 and step_col=1",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 2,
                    "disp_row": [0, 3],
                    "disp_col": [-2, 2],
                    "data_left": np.full((10, 10), 1),
                    "data_right": np.full((10, 10), 1),
                },
                (10, 10, 9, 7),  # (row, col, disp_col, disp_row)
                np.arange(0, 3.5, 0.5),  # [0, 0.5, 1, 1.5, 2, 2.5, 3]
                np.arange(-2, 2.5, 0.5),  # [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
                id="subpix=2, step_row=1 and step_col=1",
            ),
            pytest.param(
                {
                    "step": [2, 3],
                    "subpix": 2,
                    "disp_row": [0, 3],
                    "disp_col": [-2, 2],
                    "data_left": np.full((10, 10), 1),
                    "data_right": np.full((10, 10), 1),
                },
                (5, 4, 9, 7),  # (row, col, disp_col, disp_row)
                np.arange(0, 3.5, 0.5),  # [0, 0.5, 1, 1.5, 2, 2.5, 3] # step has no influence on subpix disparity range
                np.arange(-2, 2.5, 0.5),  # [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
                id="subpix=2, step_row=2 and step_col=3",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [0, 3],
                    "disp_col": [-2, 2],
                    "data_left": np.full((10, 10), 1),
                    "data_right": np.full((10, 10), 1),
                },
                (10, 10, 17, 13),  # (row, col, disp_col, disp_row)
                np.arange(0, 3.25, 0.25),  # [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
                np.arange(
                    -2, 2.25, 0.25
                ),  # [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
                id="subpix=4, step_row=1 and step_col=1",
            ),
            pytest.param(
                {
                    "step": [3, 2],
                    "subpix": 4,
                    "disp_row": [0, 3],
                    "disp_col": [-2, 2],
                    "data_left": np.full((10, 10), 1),
                    "data_right": np.full((10, 10), 1),
                },
                (4, 5, 17, 13),  # (row, col, disp_col, disp_row)
                np.arange(
                    0, 3.25, 0.25  # step has no influence on subpix disparity range
                ),  # [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
                np.arange(
                    -2, 2.25, 0.25
                ),  # [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
                id="subpix=4, step_row=3 and step_col=2",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_subpix(self, shape_expected, row_disparity, col_disparity, make_cost_volumes):
        """
        Test subpix parameter in matching cost
        """

        cost_volumes = make_cost_volumes

        # Check that the cost volume has the correct shape
        np.testing.assert_array_equal(cost_volumes["cost_volumes"].shape, shape_expected)
        # Check that the subpixel row disparities are correct
        np.testing.assert_array_equal(cost_volumes.disp_row, row_disparity)
        # Check that the subpixel col disparities are correct
        np.testing.assert_array_equal(cost_volumes.disp_col, col_disparity)

    @pytest.mark.parametrize(
        ["make_cost_volumes", "index_disp_col_zero", "index_best_disp_row"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 2,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.5, 0.5, 0.5, 0.5, 0.5],
                            [1.5, 1.5, 1.5, 1.5, 1.5],
                            [2.5, 2.5, 2.5, 2.5, 2.5],
                            [3.5, 3.5, 3.5, 3.5, 3.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                4,  # disp_col[4]=0
                np.array([[5]]),  # disp_row[5]=0.5
                id="Subpix=2 and rows shifted by 0.5",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.75, 0.75, 0.75, 0.75, 0.75],
                            [1.75, 1.75, 1.75, 1.75, 1.75],
                            [2.75, 2.75, 2.75, 2.75, 2.75],
                            [3.75, 3.75, 3.75, 3.75, 3.75],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_col[8]=0
                np.array([[9]]),  # disp_row[9]=0.25
                id="Subpix=4 and rows shifted by 0.25",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.25, 0.25, 0.25, 0.25, 0.25],
                            [1.25, 1.25, 1.25, 1.25, 1.25],
                            [2.25, 2.25, 2.25, 2.25, 2.25],
                            [3.25, 3.25, 3.25, 3.25, 3.25],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_col[8]=0
                np.array([[11]]),  # disp_row[11]=0.75
                id="Subpix=4 and rows shifted by 0.75",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_positive_row(self, make_cost_volumes, index_disp_col_zero, index_best_disp_row):
        """
        Test cost volume values when using subpix with a constant positive row shift.
        """

        cost_volumes = make_cost_volumes

        # We test that for each point of the cost volume with disp_col=0 (no shift in columns)
        # we obtain that the mininimum value corresponds to the correct disparity (the one at index_best_disp_row)
        # we also check that with a subpix different from 1 we obtain a single minimum,
        # whereas with a subpix=1 we can obtain several.

        # If the shift is positive, we test all the rows expect the last one for which the shift is equal to 0.
        for col in range(cost_volumes["cost_volumes"].shape[1]):
            for row in range(cost_volumes["cost_volumes"].shape[0] - 1):

                # index_min = all minimum value indexes
                index_min = np.where(
                    cost_volumes["cost_volumes"][row, col, index_disp_col_zero, :]
                    == cost_volumes["cost_volumes"][row, col, index_disp_col_zero, :].min()
                )
                np.testing.assert_array_equal(index_min, index_best_disp_row)

    @pytest.mark.parametrize(
        ["make_cost_volumes", "index_disp_col_zero", "index_best_disp_row"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 2,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.5, 1.5, 1.5, 1.5, 1.5],
                            [2.5, 2.5, 2.5, 2.5, 2.5],
                            [3.5, 3.5, 3.5, 3.5, 3.5],
                            [4.5, 4.5, 4.5, 4.5, 4.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                4,  # disp_col[4]=0
                np.array([[3]]),  # disp_row[3]=-0.5
                id="Subpix=2 and rows shifted by -0.5",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.25, 1.25, 1.25, 1.25, 1.25],
                            [2.25, 2.25, 2.25, 2.25, 2.25],
                            [3.25, 3.25, 3.25, 3.25, 3.25],
                            [4.25, 4.25, 4.25, 4.25, 4.25],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_col[8]=0
                np.array([[7]]),  # disp_row[7]=-0.25
                id="Subpix=4 and rows shifted by -0.25",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.75, 1.75, 1.75, 1.75, 1.75],
                            [2.75, 2.75, 2.75, 2.75, 2.75],
                            [3.75, 3.75, 3.75, 3.75, 3.75],
                            [4.75, 4.75, 4.75, 4.75, 4.75],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_col[8]=0
                np.array([[5]]),  # disp_row[5]=-0.75
                id="Subpix=4 and rows shifted by -0.75",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_negative_row(self, make_cost_volumes, index_disp_col_zero, index_best_disp_row):
        """
        Test cost volume values when using subpix with a constant negative row shift.
        """

        cost_volumes = make_cost_volumes

        # We test that for each point of the cost volume with disp_col=0 (no shift in columns)
        # we obtain that the mininimum value corresponds to the correct disparity (the one at index_best_disp_row)
        # we also check that with a subpix different from 1 we obtain a single minimum,
        # whereas with a subpix=1 we can obtain several.

        # If the shift is negative, we test all the rows expect the first one for which the shift is equal to 0.
        for col in range(cost_volumes["cost_volumes"].shape[1]):
            for row in range(1, cost_volumes["cost_volumes"].shape[0]):

                # index_min = all minimum value indexes
                index_min = np.where(
                    cost_volumes["cost_volumes"][row, col, index_disp_col_zero, :]
                    == cost_volumes["cost_volumes"][row, col, index_disp_col_zero, :].min()
                )
                np.testing.assert_array_equal(index_min, index_best_disp_row)

    @pytest.mark.parametrize(
        ["make_cost_volumes", "index_disp_row_zero", "index_best_disp_col"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 2,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                4,  # disp_row[4]=0
                np.array([[5]]),  # disp_col[5]=0.5
                id="Subpix=2 and columns shifted by 0.5",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.75, 1.75, 2.75, 3.75, 4.75],
                            [0.75, 1.75, 2.75, 3.75, 4.75],
                            [0.75, 1.75, 2.75, 3.75, 4.75],
                            [0.75, 1.75, 2.75, 3.75, 4.75],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_row[8]=0
                np.array([[9]]),  # disp_col[9]=0.25
                id="Subpix=4 and columns shifted by 0.25",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.25, 1.25, 2.25, 3.25, 4.25],
                            [0.25, 1.25, 2.25, 3.25, 4.25],
                            [0.25, 1.25, 2.25, 3.25, 4.25],
                            [0.25, 1.25, 2.25, 3.25, 4.25],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_row[8]=0
                np.array([[11]]),  # disp_col[11]=0.75
                id="Subpix=4 and columns shifted by 0.75",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_positive_col(self, make_cost_volumes, index_disp_row_zero, index_best_disp_col):
        """
        Test cost volume values when using subpix with a constant positive column shift.
        """

        cost_volumes = make_cost_volumes

        # We test that for each point of the cost volume with disp_row=0 (no shift in rows)
        # we obtain that the mininimum value corresponds to the correct disparity (the one at index_best_disp_col)
        # we also check that with a subpix different from 1 we obtain a single minimum,
        # whereas with a subpix=1 we can obtain several.

        # If the shift is positive, we test all the columns expect the last one for which the shift is equal to 0.
        for col in range(cost_volumes["cost_volumes"].shape[1] - 1):
            for row in range(cost_volumes["cost_volumes"].shape[0]):

                # index_min = all minimum value indexes
                index_min = np.where(
                    cost_volumes["cost_volumes"][row, col, :, index_disp_row_zero]
                    == cost_volumes["cost_volumes"][row, col, :, index_disp_row_zero].min()
                )
                np.testing.assert_array_equal(index_min, index_best_disp_col)

    @pytest.mark.parametrize(
        ["make_cost_volumes", "index_disp_row_zero", "index_best_disp_col"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 2,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.5, 2.5, 3.5, 4.5, 5.5],
                            [1.5, 2.5, 3.5, 4.5, 5.5],
                            [1.5, 2.5, 3.5, 4.5, 5.5],
                            [1.5, 2.5, 3.5, 4.5, 5.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                4,  # disp_row[4]=0
                np.array([[3]]),  # disp_col[3]=-0.5
                id="Subpix=2 and columns shifted by -0.5",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.25, 2.25, 3.25, 4.25, 5.25],
                            [1.25, 2.25, 3.25, 4.25, 5.25],
                            [1.25, 2.25, 3.25, 4.25, 5.25],
                            [1.25, 2.25, 3.25, 4.25, 5.25],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_row[8]=0
                np.array([[7]]),  # disp_col[7]=-0.25
                id="Subpix=4 and columns shifted by -0.25",
            ),
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 4,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.75, 2.75, 3.75, 4.75, 5.75],
                            [1.75, 2.75, 3.75, 4.75, 5.75],
                            [1.75, 2.75, 3.75, 4.75, 5.75],
                            [1.75, 2.75, 3.75, 4.75, 5.75],
                        ),
                        dtype=np.float64,
                    ),
                },
                8,  # disp_row[8]=0
                np.array([[5]]),  # disp_col[5]=-0.75
                id="Subpix=4 and columns shifted by -0.75",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_negative_col(self, make_cost_volumes, index_disp_row_zero, index_best_disp_col):
        """
        Test cost volume values when using subpix with a constant negative column shift.
        """

        cost_volumes = make_cost_volumes

        # We test that for each point of the cost volume with disp_row=0 (no shift in rows)
        # we obtain that the mininimum value corresponds to the correct disparity (the one at index_best_disp_col)
        # we also check that with a subpix different from 1 we obtain a single minimum,
        # whereas with a subpix=1 we can obtain several.

        # If the shift is negative, we test all the columns expect the first one for which the shift is equal to 0.
        for col in range(1, cost_volumes["cost_volumes"].shape[1]):
            for row in range(cost_volumes["cost_volumes"].shape[0]):

                # index_min = all minimum value indexes
                index_min = np.where(
                    cost_volumes["cost_volumes"][row, col, :, index_disp_row_zero]
                    == cost_volumes["cost_volumes"][row, col, :, index_disp_row_zero].min()
                )
                np.testing.assert_array_equal(index_min, index_best_disp_col)

    @pytest.mark.parametrize(
        ["make_cost_volumes", "row", "col", "disp_col"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 1,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [1.5, 1.5, 1.5, 1.5, 1.5],
                            [2.5, 2.5, 2.5, 2.5, 2.5],
                            [3.5, 3.5, 3.5, 3.5, 3.5],
                            [4.5, 4.5, 4.5, 4.5, 4.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                1,
                1,
                2,  # disp_col[2]=0
                id="Subpix=1 and rows shifted by -0.5",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_1_row(self, make_cost_volumes, row, col, disp_col):
        """
        Test cost volume can have several minimum when using subpix=1 with a constant shift in rows.
        """

        cost_volumes = make_cost_volumes

        # Test that we have several minimum
        # Constant shift in rows

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                len(
                    np.where(
                        cost_volumes["cost_volumes"][row, col, disp_col, :]
                        == cost_volumes["cost_volumes"][row, col, disp_col, :].min()
                    )[0]
                ),
                1,
            )

    @pytest.mark.parametrize(
        ["make_cost_volumes", "row", "col", "disp_row"],
        [
            pytest.param(
                {
                    "step": [1, 1],
                    "subpix": 1,
                    "disp_row": [-2, 2],
                    "disp_col": [-2, 2],
                    "data_left": np.array(
                        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
                        dtype=np.float64,
                    ),
                    "data_right": np.array(
                        (
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                            [0.5, 1.5, 2.5, 3.5, 4.5],
                        ),
                        dtype=np.float64,
                    ),
                },
                3,
                3,
                2,  # disp_row[2]=0
                id="Subpix=1 and columns shifted by 0.5",
            ),
        ],
        indirect=["make_cost_volumes"],
    )
    def test_cost_volumes_values_subpix_1_col(self, make_cost_volumes, row, col, disp_row):
        """
        Test cost volume can have several minimum when using subpix=1 with a constant shift in columns.
        """

        cost_volumes = make_cost_volumes

        # Test that we have several minimum
        # Constant shift in columns

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                len(
                    np.where(
                        cost_volumes["cost_volumes"][row, col, :, disp_row]
                        == cost_volumes["cost_volumes"][row, col, :, disp_row].min()
                    )[0]
                ),
                1,
            )


class TestDisparityMargins:
    """
    Test the addition of disparity margins in the cost volume
    """

    @pytest.fixture()
    def create_datasets(self):
        """
        Creates left and right datasets
        """

        data = np.full((5, 5), 1)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        add_disparity_grid(left, [0, 1], [-1, 0])

        left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "col_disparity_source": [0, 1],
            "row_disparity_source": [-1, 0],
        }

        data = np.full((5, 5), 1)
        right = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "col_disparity_source": [0, 1],
            "row_disparity_source": [-1, 0],
        }

        return left, right

    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc"])
    @pytest.mark.parametrize(
        ["margins", "subpix", "gt_cv_shape", "gt_disp_col", "gt_disp_row"],
        [
            pytest.param(
                None,
                1,
                (5, 5, 2, 2),  # margins=None -> we do not add disparity margins
                [0, 1],
                [-1, 0],
                id="Margins=None",
            ),
            pytest.param(
                Margins(0, 0, 0, 0),
                1,
                (5, 5, 2, 2),
                [0, 1],
                [-1, 0],  # margins=(0,0,0,0) -> we do not add disparity margins
                id="Margins(left=0, up=0, right=0, down=0)",
            ),
            pytest.param(
                Margins(3, 3, 3, 3),
                1,
                (
                    5,
                    5,
                    8,
                    8,
                ),
                [-3, -2, -1, 0, 1, 2, 3, 4],
                [-4, -3, -2, -1, 0, 1, 2, 3],
                # margins=(3,3,3,3) -> we add a margin of 3 on disp_min_col, disp_max_col, disp_min_row, disp_max_row
                id="Margins(left=3, up=3, right=3, down=3)",
            ),
            pytest.param(
                Margins(0, 1, 2, 3),
                1,
                (
                    5,
                    5,
                    4,
                    6,
                ),
                [0, 1, 2, 3],
                [-2, -1, 0, 1, 2, 3],
                # margins=(0,1,2,3) -> we add a margin of 0 on disp_min_col, 2 on disp_max_col,
                # 1 on disp_min_row and 3 on disp_max_row
                id="Margins(left=0, up=1, right=2, down=3)",
            ),
            pytest.param(
                Margins(4, 2, 4, 2),
                1,
                (
                    5,
                    5,
                    10,
                    6,
                ),
                [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                [-3, -2, -1, 0, 1, 2],
                # margins=(4,2,4,2) -> we add a margin of 4 on disp_min_col and on disp_max_col
                # and of 2 on disp_min_row and disp_max_row
                id="Margins(left=4, up=2, right=4, down=2)",
            ),
            pytest.param(
                Margins(2, 6, 2, 6),
                1,
                (
                    5,
                    5,
                    6,
                    14,
                ),
                [-2, -1, 0, 1, 2, 3],
                [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                # margins=(2,6,2,6) -> we add a margin of 2 on disp_min_col and on disp_max_col
                # and of 6 on disp_min_row and disp_max_row
                id="Margins(left=2, up=6, right=2, down=6)",
            ),
            pytest.param(
                Margins(6, 2, 6, 2),
                1,
                (
                    5,
                    5,
                    14,
                    6,
                ),
                [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
                [-3, -2, -1, 0, 1, 2],
                # margins=(6,2,6,2) -> we add a margin of 6 on disp_min_col and on disp_max_col
                # and of 2 on disp_min_row and disp_max_row
                id="Margins(left=6, up=2, right=6, down=2)",
            ),
            pytest.param(
                Margins(3, 3, 3, 3),
                2,
                (
                    5,
                    5,
                    15,
                    15,
                ),
                [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
                [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                # margins=(3,3,3,3) and subpix=2 -> we add a margin of 3x2 on disp_min_col, disp_max_col,
                # disp_min_row, disp_max_row
                id="Margins(left=3, up=3, right=3, down=3), subpix=2",
            ),
            pytest.param(
                Margins(0, 1, 2, 3),
                2,
                (
                    5,
                    5,
                    7,
                    11,
                ),
                [0, 0.5, 1, 1.5, 2, 2.5, 3],
                [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                # margins=(0,1,2,3) -> we add a margin of 0 on disp_min_col, 2x2 on disp_max_col,
                # 1x2 on disp_min_row and 3x2 on disp_max_row
                id="Margins(left=0, up=1, right=2, down=3)",
            ),
            pytest.param(
                Margins(6, 4, 2, 3),
                2,
                (
                    5,
                    5,
                    19,
                    17,
                ),
                [-6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                # margins=(6,4,2,3) -> we add a margin of 6x2 on disp_min_col, 2x2 on disp_max_col,
                # 4x2 on disp_min_row and 3x2 on disp_max_row
                id="Margins(left=6, up=4, right=2, down=3)",
            ),
            pytest.param(
                Margins(0, 0, 0, 0),
                4,
                (
                    5,
                    5,
                    5,
                    5,
                ),
                [0, 0.25, 0.5, 0.75, 1],
                [-1, -0.75, -0.5, -0.25, 0],  # we do not add disparity margins
                id="Margins(left=0, up=0, right=0, down=0), subpix=4",
            ),
            pytest.param(
                Margins(0, 1, 2, 3),
                4,
                (
                    5,
                    5,
                    13,
                    21,
                ),
                [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
                [
                    -2,
                    -1.75,
                    -1.5,
                    -1.25,
                    -1,
                    -0.75,
                    -0.5,
                    -0.25,
                    0,
                    0.25,
                    0.5,
                    0.75,
                    1,
                    1.25,
                    1.5,
                    1.75,
                    2,
                    2.25,
                    2.5,
                    2.75,
                    3,
                ],
                # margins=(0,1,2,3) -> we add a margin of 0 on disp_min_col, 2x4 on disp_max_col,
                # 1x4 on disp_min_row and 3x4 on disp_max_row
                id="Margins(left=0, up=1, right=2, down=3), subpix=4",
            ),
            pytest.param(
                Margins(3, 3, 3, 3),
                4,
                (
                    5,
                    5,
                    29,
                    29,
                ),
                np.arange(-3, 4.25, 0.25),
                np.arange(-4, 3.25, 0.25),
                # margins=(3,3,3,3) and subpix=4 -> we add a margin of 3x4 on disp_min_col, disp_max_col,
                # disp_min_row, disp_max_row
                id="Margins(left=3, up=3, right=3, down=3), subpix=4",
            ),
        ],
    )
    def test_compute_cost_volume_margins(
        self, create_datasets, margins, subpix, gt_cv_shape, gt_disp_col, gt_disp_row, matching_cost_method
    ):
        """
        Test the addition of margins on the disparities dimensions of the cost_volumes
        after the compute_cost_volume method
        """

        cfg = {
            "pipeline": {
                "matching_cost": {"matching_cost_method": matching_cost_method, "window_size": 1, "subpix": subpix}
            }
        }

        left, right = create_datasets

        # Initialize matching_cost
        matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

        # Allocate cost volume
        matching_cost_matcher.allocate_cost_volume_pandora(
            img_left=left,
            img_right=right,
            grid_min_col=np.full((5, 5), 0),
            grid_max_col=np.full((5, 5), 1),
            cfg=cfg,
            margins=margins,
        )

        # minimum and maximum column disparity may have been modified
        # after disparity margins were added in allocate_cost_volume_pandora
        disp_col_min = matching_cost_matcher.grid_.disp.min()
        disp_col_max = matching_cost_matcher.grid_.disp.max()

        # compute cost volumes
        cost_volumes = matching_cost_matcher.compute_cost_volumes(
            img_left=left,
            img_right=right,
            grid_min_col=np.full((5, 5), disp_col_min),
            grid_max_col=np.full((5, 5), disp_col_max),
            grid_min_row=np.full((5, 5), -1),
            grid_max_row=np.full((5, 5), 0),
            margins=margins,
        )

        np.testing.assert_array_equal(cost_volumes["cost_volumes"].shape, gt_cv_shape)
        np.testing.assert_array_equal(cost_volumes["cost_volumes"].disp_col, gt_disp_col)
        np.testing.assert_array_equal(cost_volumes["cost_volumes"].disp_row, gt_disp_row)
