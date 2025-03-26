# Copyright (c) 2025 CS GROUP France
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

from typing import Dict

# pylint: disable=redefined-outer-name, protected-access, unused-argument
# mypy: disable-error-code=attr-defined

import numpy as np
import pytest
import xarray as xr
from json_checker.core.exceptions import DictCheckerError
from pandora.margins import Margins
from pandora2d import refinement, common, matching_cost, disparity, criteria
from pandora2d.refinement.optical_flow import OpticalFlow
from pandora2d.img_tools import add_disparity_grid


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
        "col_disparity_source": [-1, 3],
        "row_disparity_source": [-1, 3],
        "invalid_disparity": np.nan,
    }

    return img


@pytest.fixture()
def optical_flow_cfg():
    return {"refinement_method": "optical_flow"}


def test_check_conf_passes(optical_flow_cfg):
    """
    Description : Test the check_conf function
    Data :
    Requirement : EX_REF_01, EX_REF_FO_00
    """
    refinement.AbstractRefinement(optical_flow_cfg)  # type: ignore[abstract]


class TestIterations:
    """
    Description : Test Iteration parameter.
    Requirement : EX_REF_FO_01
    """

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
        """
        Description : Iteration should be only positive integer.
        Data :
        Requirement : EX_CONF_08
        """
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
        """
        Description : Wrong value of window_size
        Data :
        Requirement : EX_CONF_08
        """
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
        """
        Description : Wrong value of step
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(DictCheckerError) as err:
            refinement.AbstractRefinement(optical_flow_cfg, step)  # type: ignore[abstract]
        assert "step" in err.value.args[0]


def test_reshape_to_matching_cost_window_left(dataset_image):
    """
    Test reshape_to_matching_cost_window function for a left image
    """

    img = dataset_image

    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, [1, 1], 3
    )  # type: ignore[abstract]

    cv = np.zeros((6, 5, 7, 5))

    disparity_range_row = np.arange(-2, 4 + 1)
    disparity_range_col = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_row", "disp_col"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_row": disparity_range_row,
            "disp_col": disparity_range_col,
        },
        attrs={"offset_row_col": 1},
    )

    # get first and last coordinates for row and col in cost volume dataset
    first_col_coordinate = cost_volumes.col.data[0] + cost_volumes.offset_row_col
    last_col_coordinate = cost_volumes.col.data[-1] - cost_volumes.offset_row_col
    col_extrema_coordinates = [
        OpticalFlow.find_nearest_column(first_col_coordinate, cost_volumes.col.data, "+"),
        OpticalFlow.find_nearest_column(last_col_coordinate, cost_volumes.col.data, "-"),
    ]

    first_row_coordinate = cost_volumes.row.data[0] + cost_volumes.offset_row_col
    last_row_coordinate = cost_volumes.row.data[-1] - cost_volumes.offset_row_col
    row_extrema_coordinates = [
        OpticalFlow.find_nearest_column(first_row_coordinate, cost_volumes.row.data, "+"),
        OpticalFlow.find_nearest_column(last_row_coordinate, cost_volumes.row.data, "-"),
    ]

    # for left image
    reshaped_left = refinement_class.reshape_to_matching_cost_window(
        img, cost_volumes, (row_extrema_coordinates, col_extrema_coordinates)
    )

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
    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, [1, 1], 3
    )  # type: ignore[abstract]

    # Create disparity maps
    col_disp_map = [2, 0, 0, 0, 1, 0, 0, 0, 1, -2, 0, 0]
    row_disp_map = [2, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0]

    cv = np.zeros((6, 5, 7, 5))

    disparity_range_row = np.arange(-2, 4 + 1)
    disparity_range_col = np.arange(-2, 2 + 1)

    cost_volumes = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_row", "disp_col"], cv)},
        coords={
            "row": np.arange(0, 6),
            "col": np.arange(0, 5),
            "disp_row": disparity_range_row,
            "disp_col": disparity_range_col,
        },
        attrs={"offset_row_col": 1},
    )

    # get first and last coordinates for row and col in cost volume dataset
    first_col_coordinate = cost_volumes.col.data[0] + cost_volumes.offset_row_col
    last_col_coordinate = cost_volumes.col.data[-1] - cost_volumes.offset_row_col
    col_extrema_coordinates = [
        OpticalFlow.find_nearest_column(first_col_coordinate, cost_volumes.col.data, "+"),
        OpticalFlow.find_nearest_column(last_col_coordinate, cost_volumes.col.data, "-"),
    ]

    first_row_coordinate = cost_volumes.row.data[0] + cost_volumes.offset_row_col
    last_row_coordinate = cost_volumes.row.data[-1] - cost_volumes.offset_row_col
    row_extrema_coordinates = [
        OpticalFlow.find_nearest_column(first_row_coordinate, cost_volumes.row.data, "+"),
        OpticalFlow.find_nearest_column(last_row_coordinate, cost_volumes.row.data, "-"),
    ]

    # for right image
    reshaped_right = refinement_class.reshape_to_matching_cost_window(
        img, cost_volumes, (row_extrema_coordinates, col_extrema_coordinates), row_disp_map, col_disp_map
    )

    # test four matching_cost
    idx_1_1 = [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
    idx_2_2 = [[2, 3, 4], [7, 8, 9], [12, 13, 14]]

    assert np.array_equal(reshaped_right[:, :, 0], idx_1_1)
    assert np.array_equal(reshaped_right[:, :, 4], idx_2_2)


@pytest.mark.parametrize(
    ["window_size", "mc_1", "mc_2", "gt_mc_1", "gt_mc_2"],
    [
        pytest.param(
            5,
            np.array(
                [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16], [18, 19, 20, 21, 22], [24, 25, 26, 27, 28]]
            ),
            np.array(
                [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11], [13, 14, 15, 16, 17], [19, 20, 21, 22, 23], [25, 26, 27, 28, 29]]
            ),
            np.array(
                [
                    [19, 20, 21, 22, 22],
                    [25, 26, 27, 28, 28],
                    [25, 26, 27, 28, 28],
                    [19, 20, 21, 22, 22],
                    [13, 14, 15, 16, 16],
                ]
            ),
            np.array(
                [
                    [20, 21, 22, 23, 23],
                    [26, 27, 28, 29, 29],
                    [26, 27, 28, 29, 29],
                    [20, 21, 22, 23, 23],
                    [14, 15, 16, 17, 17],
                ]
            ),
        ),
        pytest.param(
            3,
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[7, 8, 8], [4, 5, 5], [1, 2, 2]]),
            np.array([[8, 9, 9], [5, 6, 6], [2, 3, 3]]),
        ),
    ],
)
def test_warped_image_without_step(window_size, mc_1, mc_2, gt_mc_1, gt_mc_2):
    """
    test warped image with different window size
    no test for window size at 1 because "window_size": And(int, lambda input: input > 1 and (input % 2) != 0)
    """

    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, None, window_size
    )  # type: ignore[abstract]

    reshaped_right = np.stack((mc_1, mc_2)).transpose((1, 2, 0))
    delta_row = -3 * np.ones(2)
    delta_col = -np.ones(2)

    test_img_shift = refinement_class.warped_img(reshaped_right, delta_row, delta_col, [0, 1])

    # check that the generated image is equal to ground truth
    assert np.array_equal(gt_mc_1, test_img_shift[:, :, 0])
    assert np.array_equal(gt_mc_2, test_img_shift[:, :, 1])


def test_optical_flow_method():
    """
    test optical flow method with a simple col shift
    """

    # input array creation
    array_left = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    array_right = np.array(
        [
            [0.1, 1.2, 2.3, 3.4, 4.5],
            [0.1, 1.2, 2.3, 3.4, 4.5],
            [0.1, 1.2, 2.3, 3.4, 4.5],
            [0.1, 1.2, 2.3, 3.4, 4.5],
            [0.1, 1.2, 2.3, 3.4, 4.5],
        ]
    )

    one_dim_size = (array_left.shape[0] - 2) * (array_left.shape[1] - 2)  # -2 because of margin

    # patch creation
    patches_left = np.lib.stride_tricks.sliding_window_view(array_left, [3, 3])
    patches_left = patches_left.reshape((one_dim_size, 3, 3)).transpose((1, 2, 0))
    patches_right = np.lib.stride_tricks.sliding_window_view(array_right, [3, 3])
    patches_right = patches_right.reshape((one_dim_size, 3, 3)).transpose((1, 2, 0))

    idx_to_compute = np.arange(patches_left.shape[2]).tolist()

    # class initialisation
    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, [1, 1], 3
    )  # type: ignore[abstract]

    computed_drow, computed_dcol, idx_to_compute = refinement_class.optical_flow(
        patches_left, patches_right, idx_to_compute
    )

    truth_drow = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    truth_dcol = [0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4]

    assert np.allclose(computed_dcol, truth_dcol, atol=1e-03)
    assert np.allclose(computed_drow, truth_drow, atol=1e-03)


def test_lucas_kanade_core_algorithm():
    """
    test lucas kanade algorithm with simple flow in x axis
    """

    left_data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)

    right_data = np.array([[2.1, 3.1, 4.1], [2.1, 3.1, 4.1], [2.1, 3.1, 4.1]], dtype=float)

    refinement_class = refinement.AbstractRefinement({"refinement_method": "optical_flow"})  # type: ignore[abstract]
    motion_y, motion_x = refinement_class.lucas_kanade_core_algorithm(left_data, right_data)

    expected_motion = [1.1, 0.0]
    assert np.allclose([motion_x, motion_y], expected_motion, atol=1e-3)


@pytest.fixture()
def make_data(row, col):
    return np.random.uniform(0, row * col, (row, col))


def make_img_dataset(data, shift=0):
    """
    Instantiate an image dataset with specified rows, columns, and row shift.
    """
    # data = np.roll(data, shift, axis=0)
    data = data * 2.2
    data = np.round(data, 2)

    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        attrs={
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "invalid_disparity": np.nan,
        },
    ).pipe(add_disparity_grid, {"init": 0, "range": 2}, {"init": 0, "range": 2})


@pytest.fixture()
def make_left_right_images(make_data):
    data = make_data
    return make_img_dataset(data, 0), make_img_dataset(data, 2)


def make_cv_dataset(dataset_img, dataset_img_shift, cfg_mc):
    """
    Instantiate a cost volume dataset
    """
    matching_cost_matcher = matching_cost.PandoraMatchingCostMethods(cfg_mc["pipeline"]["matching_cost"])

    matching_cost_matcher.allocate(
        img_left=dataset_img,
        img_right=dataset_img_shift,
        cfg=cfg_mc,
    )

    dataset_cv = matching_cost_matcher.compute_cost_volumes(dataset_img, dataset_img_shift)
    return dataset_cv


def make_disparity_dataset(dataset_cv, cfg_disp):
    """
    Instantiate a disparity dataset
    """

    dataset_validity = criteria.get_validity_dataset(dataset_cv["criteria"])

    disparity_matcher = disparity.Disparity(cfg_disp)
    delta_x, delta_y, score = disparity_matcher.compute_disp_maps(dataset_cv)

    data_variables = {
        "row_map": (("row", "col"), delta_x),
        "col_map": (("row", "col"), delta_y),
        "correlation_score": (("row", "col"), score),
    }
    coords = {"row": dataset_cv.row.data, "col": dataset_cv.col.data}
    dataset = xr.Dataset(data_variables, coords)
    dataset_disp_map = common.dataset_disp_maps(
        dataset.row_map,
        dataset.col_map,
        dataset.coords,
        dataset.correlation_score,
        dataset_validity,
        attributes={"invalid_disp": np.nan},
    )
    return dataset_disp_map


@pytest.mark.parametrize(["row", "col", "step_row", "step_col"], [(10, 10, 2, 1), (10, 10, 1, 2), (10, 10, 2, 2)])
def test_step_with_refinement_method(make_left_right_images, row, col, step_row, step_col):
    """
    Test refinement method with a step
    """

    # create left image dataset  and right image dataset with same as left but with a row shift
    dataset_img, dataset_img_shift = make_left_right_images

    # create cost volume dataset
    cfg_mc = {
        "pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3, "step": [step_row, step_col]}}
    }
    dataset_cv = make_cv_dataset(dataset_img, dataset_img_shift, cfg_mc)

    # create disparity dataset
    cfg_disp = {"disparity_method": "wta", "invalid_disparity": np.nan}
    dataset_disp_map = make_disparity_dataset(dataset_cv, cfg_disp)

    # Start test
    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, [step_row, step_col], 3
    )  # type: ignore[abstract]

    refinement_class.refinement_method(dataset_cv, dataset_disp_map, dataset_img, dataset_img_shift)


@pytest.mark.parametrize(
    ["row", "col", "step_row", "step_col", "window_size"], [(11, 11, 1, 1, 3), (11, 11, 1, 1, 5), (11, 11, 1, 1, 11)]
)
def test_window_size_refinement_method(make_left_right_images, row, col, step_row, step_col, window_size):
    """
    Test refinement method with different windows size and check border value, here the step is fixed to 1
    """

    # create left image dataset  and right image dataset with same as left but with a row shift
    dataset_img, dataset_img_shift = make_left_right_images

    # create cost volume dataset
    cfg_mc = {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": window_size, "step": [step_row, step_col]}
        }
    }

    dataset_cv = make_cv_dataset(dataset_img, dataset_img_shift, cfg_mc)

    # create disparity dataset
    cfg_disp = {"disparity_method": "wta", "invalid_disparity": np.nan}
    dataset_disp_map = make_disparity_dataset(dataset_cv, cfg_disp)

    # Start test
    refinement_class = refinement.AbstractRefinement(
        {"refinement_method": "optical_flow"}, [step_row, step_col], window_size
    )  # type: ignore[abstract]

    delta_col, delta_row, _ = refinement_class.refinement_method(
        dataset_cv, dataset_disp_map, dataset_img, dataset_img_shift
    )

    pad = max(window_size // 2 * ele for _ in range(2) for ele in [step_row, step_col])

    # Check if there are any _invalid_disp inside image without border
    assert not np.isnan(delta_col[pad : col - pad, pad : col - pad]).any()
    assert not np.isnan(delta_row[pad : row - pad, pad : row - pad]).any()

    # Check _invalid_disp in border zone
    assert np.isnan(delta_col[0:pad, col - pad : col]).any()
    assert np.isnan(delta_row[0:pad, row - pad : row]).any()

    # Check final image shape
    assert np.array_equal(row, delta_row.shape[0])
    assert np.array_equal(row, delta_row.shape[1])
    assert np.array_equal(col, delta_col.shape[0])
    assert np.array_equal(col, delta_col.shape[1])


class TestDisparityGrids:
    """Test influence of disparity grids."""

    @pytest.fixture()
    def nb_rows(self) -> int:
        return 10

    @pytest.fixture()
    def nb_cols(self) -> int:
        return 8

    @pytest.fixture()
    def image(
        self,
        random_generator: np.random.Generator,
        nb_rows: int,
        nb_cols: int,
        min_row: bool,
        max_row: bool,
        min_col: bool,
        max_col: bool,
    ) -> xr.Dataset:
        """
        Create random image dataset with disparity grids with a range of 3 or 7.

        :param random_generator:
        :type random_generator: np.random.Generator
        :param nb_rows: number of rows in the image
        :type nb_rows: int
        :param nb_cols: number of cols in the image
        :type nb_cols: int
        :param min_row: if True, row min disparities will be a mix of 1 and 3 else will be all 1.
        :type min_row: bool
        :param max_row: if True, row max disparities will be a mix of 6 and 8 else will be all 6.
        :type max_row: bool
        :param min_col: if True, col min disparities will be a mix of 1 and 3 else will be all 1.
        :type min_col: bool
        :param max_col: if True, col max disparities will be a mix of 6 and 8 else will be all 6.
        :type max_col: bool
        :return: image dataset
        :rtype: xr.Dataset
        """
        shape = (nb_rows, nb_cols)
        data = random_generator.integers(0, 255, shape, endpoint=True)

        # disparity range must be odd and greater or equal to 5
        fixed_min = np.ones(shape)
        random_min = random_generator.choice([1, 3], shape)
        fixed_max = np.full(shape, 6)  # with min either 1 or 3 we get range 3 or 7
        random_max = random_min + 5

        row_min_disparity = random_min if min_row else fixed_min
        col_min_disparity = random_min if min_col else fixed_min
        row_max_disparity = random_max if max_row else fixed_max
        col_max_disparity = random_max if max_col else fixed_max

        return xr.Dataset(
            {
                "im": (["row", "col"], data),
                "row_disparity": (["band_disp", "row", "col"], np.array([row_min_disparity, row_max_disparity])),
                "col_disparity": (["band_disp", "row", "col"], np.array([col_min_disparity, col_max_disparity])),
            },
            coords={"row": np.arange(nb_rows), "col": np.arange(nb_cols), "band_disp": ["min", "max"]},
            attrs={
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "col_disparity_source": [np.min(col_min_disparity), np.max(col_max_disparity)],
                "row_disparity_source": [np.min(row_min_disparity), np.max(col_max_disparity)],
            },
        )

    @pytest.fixture()
    def cfg(self) -> Dict:
        return {
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "ssd",
                    "window_size": 3,
                    "step": [1, 1],
                    "subpix": 1,
                }
            }
        }

    @pytest.fixture()
    def invalid_value(self) -> int:
        return -99

    @pytest.fixture()
    def disparities(self, image: xr.Dataset, cfg: Dict, invalid_value) -> Dict:
        """Execute refinement method and return disparities."""
        matching_cost_ = matching_cost.PandoraMatchingCostMethods(cfg["pipeline"]["matching_cost"])

        matching_cost_.allocate(
            img_left=image,
            img_right=image,
            cfg=cfg,
        )

        cost_volumes = matching_cost_.compute_cost_volumes(
            img_left=image,
            img_right=image,
        )

        dataset_validity = criteria.get_validity_dataset(cost_volumes["criteria"])

        disparity_matcher = disparity.Disparity({"disparity_method": "wta", "invalid_disparity": invalid_value})

        disp_map_col, disp_map_row, correlation_score = disparity_matcher.compute_disp_maps(cost_volumes)

        data_variables = {
            "row_map": (("row", "col"), disp_map_row),
            "col_map": (("row", "col"), disp_map_col),
            "correlation_score": (("row", "col"), correlation_score),
        }

        coords = {"row": image.coords["row"], "col": image.coords["col"]}

        dataset = xr.Dataset(data_variables, coords)

        dataset_disp_map = common.dataset_disp_maps(
            dataset.row_map,
            dataset.col_map,
            dataset.coords,
            dataset.correlation_score,
            dataset_validity,
            attributes={"invalid_disp": invalid_value},
        )

        test = refinement.AbstractRefinement(
            {"refinement_method": "optical_flow"},
            cfg["pipeline"]["matching_cost"]["step"],
            cfg["pipeline"]["matching_cost"]["window_size"],
        )  # type: ignore[abstract]
        disparity_col, disparity_row, _ = test.refinement_method(cost_volumes, dataset_disp_map, image, image)
        return {"row_disparity": disparity_row, "col_disparity": disparity_col}

    @pytest.mark.parametrize("min_row", (True, False))
    @pytest.mark.parametrize("max_row", (True, False))
    @pytest.mark.parametrize("min_col", (True, False))
    @pytest.mark.parametrize("max_col", (True, False))
    def test_variable_grid(self, image, disparities, invalid_value):
        """Test resulting disparities are in range defined by grids."""
        # We want to exclude invalid_values from the comparaison
        valid_row_mask = disparities["row_disparity"] != invalid_value
        valid_col_mask = disparities["col_disparity"] != invalid_value

        assert np.all(
            disparities["row_disparity"][valid_row_mask]
            >= image["row_disparity"].sel({"band_disp": "min"}).data[valid_row_mask]
        )
        assert np.all(
            disparities["col_disparity"][valid_col_mask]
            >= image["col_disparity"].sel({"band_disp": "min"}).data[valid_col_mask]
        )
        assert np.all(
            disparities["row_disparity"][valid_row_mask]
            <= image["row_disparity"].sel({"band_disp": "max"}).data[valid_row_mask]
        )
        assert np.all(
            disparities["col_disparity"][valid_col_mask]
            <= image["col_disparity"].sel({"band_disp": "max"}).data[valid_col_mask]
        )
