#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Test configuration
"""

import pytest
import transitions
import numpy as np
import xarray as xr
from json_checker import DictCheckerError, MissKeyCheckerError
from skimage.io import imsave

from pandora2d.img_tools import create_datasets_from_inputs, add_disparity_grid
from pandora2d import check_configuration


class TestCheckDatasets:
    """Test check_datasets function."""

    @pytest.fixture()
    def datasets(self, left_img_path, right_img_path):
        """Build dataset."""
        input_cfg = {
            "left": {"img": left_img_path, "nodata": -9999},
            "right": {"img": right_img_path, "nodata": -9999},
            "col_disparity": [-2, 2],
            "row_disparity": [-3, 3],
        }
        return create_datasets_from_inputs(input_cfg)

    def test_nominal(self, datasets):
        """
        Test the nominal case with image dataset
        """
        dataset_left, dataset_right = datasets
        check_configuration.check_datasets(dataset_left, dataset_right)

    def test_fails_with_wrong_dimension(self):
        """
        Test with wrong image shapes
        """
        data_left = np.full((3, 3), 2)
        data_right = np.full((4, 4), 2)

        attributs = {
            "no_data_img": 0,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": None,
        }

        dataset_left = xr.Dataset(
            {"im": (["row", "col"], data_left)},
            coords={"row": np.arange(data_left.shape[0]), "col": np.arange(data_left.shape[1])},
            attrs=attributs,
        ).pipe(add_disparity_grid, [0, 1], [-1, 0])

        dataset_right = xr.Dataset(
            {"im": (["row", "col"], data_right)},
            coords={"row": np.arange(data_right.shape[0]), "col": np.arange(data_right.shape[1])},
            attrs=attributs,
        ).pipe(add_disparity_grid, [-2, 2], [-3, 3])

        with pytest.raises(ValueError) as exc_info:
            check_configuration.check_datasets(dataset_left, dataset_right)
        assert str(exc_info.value) == "left and right datasets must have the same shape"

    @pytest.mark.parametrize(
        ["col_disparity", "row_disparity"],
        [
            pytest.param(True, False, id="Remove col_disparity"),
            pytest.param(False, True, id="Remove row_disparity"),
            pytest.param(True, True, id="Remove col & row disparity"),
        ],
    )
    def test_fails_without_disparity(self, datasets, col_disparity, row_disparity):
        """
        Test with missing disparities
        """
        dataset_left, dataset_right = datasets
        if col_disparity:
            dataset_left = dataset_left.drop_vars("col_disparity")
        if row_disparity:
            dataset_left = dataset_left.drop_vars("row_disparity")

        with pytest.raises(ValueError) as exc_info:
            check_configuration.check_datasets(dataset_left, dataset_right)
        assert str(exc_info.value) == "left dataset must have column and row disparities DataArrays"


class TestCheckInputSection:
    """
    Test check input section.
    """

    def test_check_nominal_case(self, correct_input_cfg) -> None:
        assert check_configuration.check_input_section(correct_input_cfg)

    def test_false_input_disp_should_exit(self, false_input_disp):
        with pytest.raises(ValueError, match="disp_max must be bigger than disp_min"):
            check_configuration.check_input_section(false_input_disp)

    def test_false_input_path_image_should_raise_error(self, false_input_path_image):
        with pytest.raises(DictCheckerError):
            check_configuration.check_input_section(false_input_path_image)


class TestCheckPipelineSection:
    """Test check_pipeline_section."""

    def test_nominal_case(self, pandora2d_machine, correct_pipeline) -> None:
        """
        Test function for checking user pipeline section
        """
        assert check_configuration.check_pipeline_section(correct_pipeline, pandora2d_machine)

    def test_false_mc_dict_should_raise_error(self, pandora2d_machine, false_pipeline_mc):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(false_pipeline_mc, pandora2d_machine)

    def test_false_disp_dict_should_raise_error(self, pandora2d_machine, false_pipeline_disp):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(false_pipeline_disp, pandora2d_machine)


class TestCheckRoiSection:
    """Test check_roi_section."""

    def test_expect_roi_section(self):
        with pytest.raises(MissKeyCheckerError, match="ROI"):
            check_configuration.check_roi_section({"input": {}})

    def test_nominal_case(self, correct_roi_sensor) -> None:
        """
        Test function for checking user ROI section
        """
        # with a correct ROI check_roi_section should return nothing
        assert check_configuration.check_roi_section(correct_roi_sensor)

    def test_dimension_lt_0_raises_exception(self, false_roi_sensor_negative):
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(false_roi_sensor_negative)

    def test_first_dimension_gt_last_dimension_raises_exception(self, false_roi_sensor_first_superior_to_last):
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(false_roi_sensor_first_superior_to_last)

    @pytest.mark.parametrize(
        "roi_section",
        [
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 10},
                        "row": {"first": 10, "last": 100},
                    },
                },
                id="Only col",
            ),
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 100},
                        "row": {"first": 10, "last": 10},
                    },
                },
                id="Only row",
            ),
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 10},
                        "row": {"first": 10, "last": 10},
                    },
                },
                id="Both row and col",
            ),
        ],
    )
    def test_one_pixel_roi_is_valid(self, roi_section):
        """Should not raise error."""
        check_configuration.check_roi_section(roi_section)


def test_get_roi_pipeline(
    correct_roi_sensor,
) -> None:
    """
    Test get_roi_pipeline function
    """
    assert correct_roi_sensor == check_configuration.get_roi_config(correct_roi_sensor)


class TestCheckRoiCoherence:
    """Test check_roi_coherence."""

    def test_first_lt_last_is_ok(self, correct_roi_sensor) -> None:
        check_configuration.check_roi_coherence(correct_roi_sensor["ROI"]["col"])

    def test_first_gt_last_raises_error(self, false_roi_sensor_first_superior_to_last):
        with pytest.raises(ValueError) as exc_info:
            check_configuration.check_roi_coherence(false_roi_sensor_first_superior_to_last["ROI"]["col"])
        assert str(exc_info.value) == 'In ROI "first" should be lower than "last" in sensor ROI'


class TestCheckStep:
    """Test check_step."""

    def test_nominal_case(self, pipeline_config, pandora2d_machine) -> None:
        """
        Test step configuration with user configuration dictionary
        """

        # Add correct step
        pipeline_config["pipeline"]["matching_cost"]["step"] = [1, 1]
        assert check_configuration.check_pipeline_section(pipeline_config, pandora2d_machine)

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param([1], id="one size list"),
            pytest.param([-1, 1], id="negative value"),
            pytest.param([1, 1, 1], id="More than 2 elements"),
            pytest.param([1, "1"], id="String element"),
        ],
    )
    def test_fails_with_bad_step_values(self, pipeline_config, pandora2d_machine, step) -> None:
        """Test check_pipeline_section fails with bad values of step."""
        pipeline_config["pipeline"]["matching_cost"]["step"] = step
        with pytest.raises(DictCheckerError):
            check_configuration.check_pipeline_section(pipeline_config, pandora2d_machine)


class TestCheckConfMatchingCostNodataCondition:
    """Test check conf for right image’s nodata."""

    @pytest.fixture()
    def configuration(self, right_nodata, matching_cost_method, left_img_path, right_img_path):
        return {
            "input": {
                "left": {
                    "img": left_img_path,
                    "nodata": "NaN",
                },
                "right": {
                    "img": right_img_path,
                    "nodata": right_nodata,
                },
                "col_disparity": [-2, 2],
                "row_disparity": [-2, 2],
            },
            "pipeline": {
                "matching_cost": {"matching_cost_method": matching_cost_method, "window_size": 1},
            },
        }

    @pytest.mark.parametrize("right_nodata", ["NaN", 0.1, "inf", None])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd"])
    def test_sad_or_ssd_fail_with(self, pandora2d_machine, configuration):
        """Right nodata must be an integer with sad or ssd matching_cost_method."""
        with pytest.raises((ValueError, DictCheckerError)):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("right_nodata", [432])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc"])
    def test_passes_with_int(self, pandora2d_machine, configuration):
        """Right nodata must be an integer."""
        check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("right_nodata", ["NaN", "inf"])
    @pytest.mark.parametrize("matching_cost_method", ["zncc"])
    def test_zncc_passes_with(self, pandora2d_machine, configuration):
        """Right nodata can be inf or nan with zncc matching_cost_method."""
        check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize("right_nodata", [0.2, None])
    @pytest.mark.parametrize("matching_cost_method", ["zncc"])
    def test_zncc_fails_with(self, pandora2d_machine, configuration):
        """Right nodata must can not be float or nan with zncc matching_cost_method."""
        with pytest.raises((ValueError, DictCheckerError)):
            check_configuration.check_conf(configuration, pandora2d_machine)


class TestCheckDisparityRangeSize:
    """Test check_disparity_range_size method."""

    @pytest.mark.parametrize(
        ["disparity", "title", "string_match"],
        [
            pytest.param(
                [-1, 1],
                "Column",
                "Column disparity range with a size < 5 are not allowed with interpolation refinement method",
                id="Column disparity range < 5",
            ),
            pytest.param(
                [-3, -1],
                "Row",
                "Row disparity range with a size < 5 are not allowed with interpolation refinement method",
                id="Row disparity range < 5",
            ),
        ],
    )
    def test_fails_with_disparity_ranges_lower_5(self, disparity, title, string_match):
        """Disparity range size must be greater than or equal to 5 when interpolation is used as refinement method"""
        with pytest.raises(ValueError, match=string_match):
            check_configuration.check_disparity_range_size(disparity, title)

    @pytest.mark.parametrize(
        ["disparity", "title", "string_match"],
        [
            pytest.param(
                "disparity_grid_test",
                "Column",
                "Grid disparities are not yet handled by Pandora2D",
                id="Grid disparity",
            ),
        ],
    )
    def test_fails_with_grid_disparity(self, disparity, title, string_match):
        """Disparity grid is not handled yet by Pandora2D"""
        with pytest.raises(TypeError, match=string_match):
            check_configuration.check_disparity_range_size(disparity, title)

    @pytest.mark.parametrize(
        ["disparity", "title"],
        [
            pytest.param([-2, 2], "Col", id="Column disparity range greater than or equal to 5"),
            pytest.param([1, 5], "Row", id="Row disparity range greater than or equal to 5"),
        ],
    )
    def test_passes_with_disparity_ranges_equal_5(self, disparity, title):
        """Disparity range size is correct"""

        check_configuration.check_disparity_range_size(disparity, title)


class TestDisparityRangeAgainstImageSize:
    """Test that out of image disparity ranges are not allowed."""

    @pytest.fixture()
    def image_path(self, tmp_path):
        path = tmp_path / "tiff_file.tif"
        imsave(path, np.empty((450, 450)))
        return path

    @pytest.fixture()
    def row_disparity(self):
        return [-4, 1]

    @pytest.fixture()
    def col_disparity(self):
        return [-3, 2]

    @pytest.fixture()
    def configuration(self, image_path, row_disparity, col_disparity):
        return {
            "input": {
                "left": {
                    "img": str(image_path),
                    "nodata": "NaN",
                },
                "right": {
                    "img": str(image_path),
                    "nodata": "NaN",
                },
                "row_disparity": row_disparity,
                "col_disparity": col_disparity,
            },
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 1},
            },
        }

    @pytest.mark.parametrize(
        "row_disparity",
        [
            pytest.param([-460, -451], id="Out on left"),
            pytest.param([451, 460], id="Out on right"),
        ],
    )
    def test_row_disparity_totally_out(self, pandora2d_machine, configuration):
        """Totally out disparities should raise an error."""
        with pytest.raises(ValueError, match="Row disparity range out of image"):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize(
        "col_disparity",
        [
            pytest.param([-460, -451], id="Out on top"),
            pytest.param([451, 460], id="Out on bottom"),
        ],
    )
    def test_column_disparity_totally_out(self, pandora2d_machine, configuration):
        """Totally out disparities should raise an error."""
        with pytest.raises(ValueError, match="Column disparity range out of image"):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity"],
        [
            pytest.param([-460, -450], [100, 200], id="Partially out on left"),
            pytest.param([450, 460], [100, 200], id="Partially out on right"),
            pytest.param([100, 200], [-460, -450], id="Partially out on top"),
            pytest.param([100, 200], [450, 460], id="Partially out on bottom"),
        ],
    )
    def test_disparity_partially_out(self, pandora2d_machine, configuration):
        """Partially out should not raise error."""
        check_configuration.check_conf(configuration, pandora2d_machine)
