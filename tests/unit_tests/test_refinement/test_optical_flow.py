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
Test refinement step
"""

# pylint: disable=redefined-outer-name, protected-access
# mypy: disable-error-code=attr-defined

import numpy as np
import pytest
import xarray as xr
from json_checker.core.exceptions import DictCheckerError
from pandora.margins import Margins
from pandora2d import refinement


@pytest.fixture()
def dataset_image():
    """
    Create an image dataset
    """
    data = np.arange(30).reshape((6, 5))

    img = xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    img.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "col_disparity_source": [-2, 2],
        "row_disparity_source": [-2, 2],
        "invalid_disparity": np.nan,
    }

    return img


@pytest.fixture()
def optical_flow_cfg():
    return {"refinement_method": "optical_flow"}


def test_check_conf_passes(optical_flow_cfg):
    """
    Test the check_conf function
    """
    refinement.AbstractRefinement(optical_flow_cfg)  # type: ignore[abstract]


class TestIterations:
    """Test Iteration parameter."""

    def test_iterations_is_not_mandatory(self):
        """Should not raise error."""
        refinement.optical_flow.OpticalFlow({"refinement_method": "optical_flow"})

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(1.5, id="float"),
            pytest.param(-1, id="negative"),
            pytest.param(0, id="null"),
        ],
    )
    def test_fails_with_invalid_iteration_value(self, value):
        """Iteration should be only positive integer."""
        with pytest.raises((KeyError, DictCheckerError)):
            refinement.optical_flow.OpticalFlow({"refinement_method": "optical_flow", "iterations": value})


class TestMargins:
    """Test margins."""

    # pylint:disable=too-few-public-methods

    @pytest.mark.parametrize(
        ["window_size", "step", "margins"],
        [
            pytest.param(5, [1, 1], Margins(2, 2, 2, 2), id="default value without step"),
            pytest.param(5, [2, 1], Margins(4, 2, 4, 2), id="default value with row step"),
            pytest.param(5, [1, 2], Margins(2, 4, 2, 4), id="default value with col step"),
            pytest.param(5, [3, 2], Margins(6, 4, 6, 4), id="default value with row and col step"),
            pytest.param(9, [1, 1], Margins(4, 4, 4, 4), id="other value without step"),
            pytest.param(9, [2, 1], Margins(8, 4, 8, 4), id="other value with row step"),
            pytest.param(9, [1, 3], Margins(4, 12, 4, 12), id="other value with col step"),
            pytest.param(9, [2, 3], Margins(8, 12, 8, 12), id="other value with row and col step"),
        ],
    )
    def test_margins(self, optical_flow_cfg, window_size, step, margins):
        """
        test margins for optical flow method
        """
        _refinement = refinement.AbstractRefinement(optical_flow_cfg, step, window_size)  # type: ignore[abstract]

        assert _refinement.margins == margins, "Not a cubic kernel Margins"


class TestWindowSize:
    """Test window_size parameter."""

    @pytest.mark.parametrize(
        "window_size",
        [
            pytest.param(5, id="default value"),
            pytest.param(9, id="odd value"),
        ],
    )
    def test_nominal_case(self, optical_flow_cfg, window_size):
        """Nominal value of window_size"""
        refinement.AbstractRefinement(optical_flow_cfg, None, window_size)  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "window_size",
        [
            pytest.param(1, id="value too small"),
            pytest.param(2, id="even value"),
            pytest.param(1.0, id="float value"),
            pytest.param(-1, id="negative value"),
            pytest.param("5", id="string value"),
            pytest.param([5], id="list value"),
            pytest.param({"window_size": 5}, id="dict value"),
        ],
    )
    def test_check_conf_fails_with_wrong_window_size(self, optical_flow_cfg, window_size):
        """Wrong value of window_size"""
        with pytest.raises(DictCheckerError) as err:
            refinement.AbstractRefinement(optical_flow_cfg, [1, 1], window_size)  # type: ignore[abstract]
        assert "window_size" in err.value.args[0]


class TestStep:
    """Test step parameter."""

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(None, id="default value"),
            pytest.param([1, 1], id="default step"),
            pytest.param([2, 1], id="correct step"),
        ],
    )
    def test_nominal_case(self, optical_flow_cfg, step):
        """Nominal value of step"""
        refinement.AbstractRefinement(optical_flow_cfg, step)  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param([-2, 3], id="negative"),
            pytest.param([2], id="one element list"),
            pytest.param([2, 2, 3], id="more than two element list"),
            pytest.param(["2", 3], id="with string element"),
            pytest.param("5", id="string value"),
            pytest.param(2, id="one value"),
        ],
    )
    def test_check_conf_fails_with_wrong_step(self, optical_flow_cfg, step):
        """Wrong value of step"""
        with pytest.raises(DictCheckerError) as err:
            refinement.AbstractRefinement(optical_flow_cfg, step)  # type: ignore[abstract]
        assert "step" in err.value.args[0]


def test_reshape_to_matching_cost_window_left(dataset_image):
    """
    Test reshape_to_matching_cost_window function for a left image
    """

    img = dataset_image

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]
    refinement_class._window_size = 3

    cv = np.zeros((6, 5, 5, 5))

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_col": disparity_range_col,
            "disp_row": disparity_range_row,
        },
    )

    # for left image
    reshaped_left = refinement_class.reshape_to_matching_cost_window(img, cost_volumes)

    # test four matching_cost
    idx_1_1 = [[0, 1, 2], [5, 6, 7], [10, 11, 12]]
    idx_2_2 = [[6, 7, 8], [11, 12, 13], [16, 17, 18]]
    idx_3_3 = [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
    idx_4_1 = [[15, 16, 17], [20, 21, 22], [25, 26, 27]]

    assert np.array_equal(reshaped_left[:, :, 0], idx_1_1)
    assert np.array_equal(reshaped_left[:, :, 4], idx_2_2)
    assert np.array_equal(reshaped_left[:, :, 8], idx_3_3)
    assert np.array_equal(reshaped_left[:, :, 9], idx_4_1)


def test_reshape_to_matching_cost_window_right(dataset_image):
    """
    Test reshape_to_matching_cost_window function for a right image
    """

    img = dataset_image

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]
    refinement_class._window_size = 3

    # Create disparity maps
    col_disp_map = [2, 0, 0, 0, 1, 0, 0, 0, 1, -2, 0, 0]
    row_disp_map = [2, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0]

    cv = np.zeros((6, 5, 5, 5))

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_col": disparity_range_col,
            "disp_row": disparity_range_row,
        },
    )

    # for right image
    reshaped_right = refinement_class.reshape_to_matching_cost_window(img, cost_volumes, row_disp_map, col_disp_map)

    # test four matching_cost
    idx_1_1 = [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
    idx_2_2 = [[2, 3, 4], [7, 8, 9], [12, 13, 14]]

    assert np.array_equal(reshaped_right[:, :, 0], idx_1_1)
    assert np.array_equal(reshaped_right[:, :, 4], idx_2_2)


def test_warped_image_without_step():
    """
    test warped image
    """

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]

    mc_1 = np.array(
        [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16], [18, 19, 20, 21, 22], [24, 25, 26, 27, 28]]
    )
    mc_2 = np.array(
        [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11], [13, 14, 15, 16, 17], [19, 20, 21, 22, 23], [25, 26, 27, 28, 29]]
    )

    reshaped_right = np.stack((mc_1, mc_2)).transpose((1, 2, 0))

    delta_row = -3 * np.ones(2)
    delta_col = -np.ones(2)

    test_img_shift = refinement_class.warped_img(reshaped_right, delta_row, delta_col, [0, 1])

    gt_mc_1 = np.array(
        [[19, 20, 21, 22, 22], [25, 26, 27, 28, 28], [25, 26, 27, 28, 28], [19, 20, 21, 22, 22], [13, 14, 15, 16, 16]]
    )

    gt_mc_2 = np.array(
        [[20, 21, 22, 23, 23], [26, 27, 28, 29, 29], [26, 27, 28, 29, 29], [20, 21, 22, 23, 23], [14, 15, 16, 17, 17]]
    )

    # check that the generated image is equal to ground truth
    assert np.array_equal(gt_mc_1, test_img_shift[:, :, 0])
    assert np.array_equal(gt_mc_2, test_img_shift[:, :, 1])
