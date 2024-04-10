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
import sys

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
import xarray as xr
from pandora import import_plugin
from rasterio import Affine

from pandora2d import matching_cost
from pandora2d.img_tools import create_datasets_from_inputs


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


@pytest.mark.plugin_tests
@pytest.mark.skipif("mc_cnn" not in sys.modules, reason="MCCNN plugin not installed")
def test_compute_cv_mc_cnn():
    """
    Test the  cost volume product by mccnn
    """

    import_plugin()

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
            np.arange(1, 8),
            id="ROI and no step",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [2, 2],
            np.arange(1, 8, 2),
            np.arange(1, 8, 2),
            id="ROI and columns step_col=margins",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [3, 3, 3, 3]},
            [3, 2],
            np.arange(1, 9, 2),
            np.arange(0, 8, 3),
            id="ROI and columns step_col < margins",
        ),
        pytest.param(
            {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
            [4, 3],
            np.arange(3, 8, 3),
            np.arange(1, 8, 4),
            id="ROI and columns step_col > margins",
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
    # For the moment, row coordinates are only calculated with step value.
    # After ticket 108, margins will also be taken into account
    np.testing.assert_array_equal(cost_volumes_with_roi["cost_volumes"].coords["row"], row_expected)


@pytest.mark.parametrize(
    ["step", "col_expected", "row_expected"],
    [
        pytest.param(
            [1, 1],
            np.arange(10),
            np.arange(10),
            id="No ROI and no step",
        ),
        pytest.param(
            [2, 2],
            np.arange(0, 10, 2),
            np.arange(0, 10, 2),
            id="No ROI and step=2",
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