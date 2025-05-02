#!/usr/bin/env python
# coding: utf8
#
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
Test configuration
"""

import random
import string
import pytest
import transitions
import numpy as np
import xarray as xr
from json_checker import DictCheckerError, MissKeyCheckerError
from skimage.io import imsave

from pandora.img_tools import get_metadata
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
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 3},
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
        Description : Test with wrong image shapes
        Data :
        Requirement : EX_CONF_11
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
        ).pipe(add_disparity_grid, {"init": -1, "range": 2}, {"init": -1, "range": 3})

        dataset_right = xr.Dataset(
            {"im": (["row", "col"], data_right)},
            coords={"row": np.arange(data_right.shape[0]), "col": np.arange(data_right.shape[1])},
            attrs=attributs,
        ).pipe(add_disparity_grid, {"init": 1, "range": 2}, {"init": 1, "range": 3})

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
        Description : Test with missing disparities
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
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

    @pytest.fixture()
    def basic_estimation_cfg(self):
        return {"estimation_method": "phase_cross_correlation"}

    def test_check_nominal_case(self, correct_input_cfg) -> None:
        assert check_configuration.check_input_section(correct_input_cfg)

    def test_fails_if_input_section_is_missing(self):
        """
        Description : Test if input section is missing in the configuration file
        Data :
        Requirement : EX_CONF_01
        """
        with pytest.raises(KeyError, match="input key is missing"):
            check_configuration.check_input_section({})

    def test_false_input_path_image_should_raise_error(self, false_input_path_image):
        """
        Description : Test raises an error if the image path isn't correct
        Data : cones/monoband/right.png
        Requirement : EX_CONF_09
        """
        with pytest.raises(DictCheckerError):
            check_configuration.check_input_section(false_input_path_image)

    def test_fails_with_images_of_different_sizes(self, correct_input_cfg, make_empty_image):
        """
        Description : Images must have the same shape and size.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_11
        """
        correct_input_cfg["input"]["left"]["img"] = str(make_empty_image("left.tiff"))
        correct_input_cfg["input"]["right"]["img"] = str(make_empty_image("right.tiff", shape=(50, 50)))

        with pytest.raises(AttributeError, match="Images must have the same size"):
            check_configuration.check_input_section(correct_input_cfg)

    def test_default_nodata(self, correct_input_cfg):
        """
        Description : Default nodata value shoud be -9999.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_04
        """
        del correct_input_cfg["input"]["left"]["nodata"]

        result = check_configuration.check_input_section(correct_input_cfg)

        assert result["input"]["left"]["nodata"] == -9999
        assert result["input"]["right"]["nodata"] == -9999

    def test_check_nominal_case_with_estimation_config(self, correct_input_cfg, basic_estimation_cfg):
        """Default estimation_config value : basic config."""

        del correct_input_cfg["input"]["col_disparity"]
        del correct_input_cfg["input"]["row_disparity"]
        assert check_configuration.check_input_section(correct_input_cfg, basic_estimation_cfg)

    def test_estimation_config_with_disparity(self, correct_input_cfg, basic_estimation_cfg):
        """Default basic estimation config with disparity in user configuration."""
        with pytest.raises(
            KeyError,
            match="When using estimation, "
            "the col_disparity and row_disparity keys must not be given in the configuration file",
        ):
            check_configuration.check_input_section(correct_input_cfg, basic_estimation_cfg)


class TestCheckPipelineSection:
    """Test check_pipeline_section."""

    def test_fails_if_pipeline_section_is_missing(self, pandora2d_machine) -> None:
        """
        Description : Test if the pipeline section is missing in the configuration file
        Data :
        Requirement : EX_CONF_02
        """
        with pytest.raises(KeyError, match="pipeline key is missing"):
            assert check_configuration.check_pipeline_section({}, pandora2d_machine)

    def test_nominal_case(self, pandora2d_machine, correct_pipeline) -> None:
        """
        Description : Test function for checking user pipeline section
        Data :
        Requirement : EX_REF_00
        """
        assert check_configuration.check_pipeline_section(correct_pipeline, pandora2d_machine)

    def test_false_mc_dict_should_raise_error(self, pandora2d_machine, false_pipeline_mc):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(false_pipeline_mc, pandora2d_machine)

    def test_false_disp_dict_should_raise_error(self, pandora2d_machine, false_pipeline_disp):
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(false_pipeline_disp, pandora2d_machine)

    @pytest.mark.parametrize(
        "step_order",
        [
            ["disparity", "matching_cost", "refinement"],
            ["matching_cost", "refinement", "disparity"],
            ["matching_cost", "estimation", "disparity"],
        ],
    )
    def test_wrong_order_should_raise_error(self, pandora2d_machine, step_order):
        """
        Description : Pipeline section order is important.
        Data :
        Requirement : EX_CONF_07
        """
        steps = {
            "estimation": {"estimated_shifts": [-0.5, 1.3], "error": [1.0], "phase_diff": [1.0]},
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "dichotomy", "filter": {"method": "bicubic"}, "iterations": 2},
        }
        configuration = {"pipeline": {step: steps[step] for step in step_order}}
        with pytest.raises(transitions.core.MachineError):
            check_configuration.check_pipeline_section(configuration, pandora2d_machine)

    def test_multiband_pipeline(self, pandora2d_machine, left_rgb_path, right_rgb_path):
        """
        Description : Test the method check_conf for multiband images
        Data :
        - Left image : cones/multibands/left.tif
        - Right image : cones/multibands/right.tif
        Requirement : EX_CONF_12
        """
        input_multiband_cfg = {
            "left": {
                "img": left_rgb_path,
            },
            "right": {
                "img": right_rgb_path,
            },
            "col_disparity": {"init": -30, "range": 30},
            "row_disparity": {"init": -30, "range": 30},
        }
        cfg = {
            "input": input_multiband_cfg,
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "disparity": {"disparity_method": "wta"},
            },
            "output": {"path": "here"},
        }

        check_configuration.check_conf(cfg, pandora2d_machine)


class TestCheckOutputSection:
    """Test check_output_section"""

    def test_path_is_mandatory(self):
        with pytest.raises(MissKeyCheckerError, match="path"):
            check_configuration.check_output_section({})

    @pytest.mark.parametrize("format_", ["tiff"])
    def test_accept_optional_format(self, format_):
        check_configuration.check_output_section({"path": "/home/me/out", "format": format_})

    @pytest.mark.parametrize("format_", ["unknown"])
    def test_fails_with_bad_format(self, format_):
        with pytest.raises(DictCheckerError, match="format"):
            check_configuration.check_output_section({"path": "/home/me/out", "format": format_})


class TestGetOutputConfig:
    """Test get_output_config."""

    def test_raise_error_on_missing_output_key(self):
        with pytest.raises(MissKeyCheckerError, match="output"):
            check_configuration.get_output_config({})

    def test_default_values(self):
        result = check_configuration.get_output_config({"output": {"path": "somewhere"}})
        assert result["format"] == "tiff"

    @pytest.mark.parametrize(["key", "value"], [("format", "something")])
    def test_default_override(self, key, value):
        result = check_configuration.get_output_config({"output": {"path": "somewhere", key: value}})
        assert result[key] == value


class TestCheckRoiSection:
    """
    Description : Test check_roi_section.
    Requirement : EX_ROI_04
    """

    def test_expect_roi_section(self):
        """
        Description : Test if ROI section is missing
        Data :
        Requirement : EX_ROI_05
        """
        with pytest.raises(MissKeyCheckerError, match="ROI"):
            check_configuration.check_roi_section({"input": {}})

    def test_nominal_case(self, correct_roi_sensor) -> None:
        """
        Test function for checking user ROI section
        """
        # with a correct ROI check_roi_section should return nothing
        assert check_configuration.check_roi_section(correct_roi_sensor)

    def test_dimension_lt_0_raises_exception(self, false_roi_sensor_negative):
        """
        Description : Raises an exception if the ROI dimensions are lower than 0
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(BaseException):
            check_configuration.check_roi_section(false_roi_sensor_negative)

    def test_first_dimension_gt_last_dimension_raises_exception(self, false_roi_sensor_first_superior_to_last):
        """
        Description : Test if the first dimension of the ROI is greater than the last one
        Data :
        Requirement : EX_CONF_08
        """
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
    """
    Description : Test check_roi_coherence.
    Requirement : EX_ROI_04
    """

    def test_first_lt_last_is_ok(self, correct_roi_sensor) -> None:
        check_configuration.check_roi_coherence(correct_roi_sensor["ROI"]["col"])

    def test_first_gt_last_raises_error(self, false_roi_sensor_first_superior_to_last):
        """
        Description : Test if 'first' is greater than 'last' in ROI
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(ValueError) as exc_info:
            check_configuration.check_roi_coherence(false_roi_sensor_first_superior_to_last["ROI"]["col"])
        assert str(exc_info.value) == '"first" should be lower than "last" in sensor ROI'


class TestCheckStep:
    """
    Description : Test check_step.
    Requirement : EX_STEP_02
    """

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
        """
        Description : Test check_pipeline_section fails with bad values of step.
        Data :
        Requirement : EX_CONF_08
        """
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
                "col_disparity": {"init": 1, "range": 2},
                "row_disparity": {"init": 1, "range": 2},
            },
            "pipeline": {
                "matching_cost": {"matching_cost_method": matching_cost_method, "window_size": 1},
            },
            "output": {"path": "there"},
        }

    @pytest.mark.parametrize("right_nodata", ["NaN", 0.1, "inf", None])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd"])
    def test_sad_or_ssd_fail_with(self, pandora2d_machine, configuration):
        """
        Description : Right nodata must be an integer with sad or ssd matching_cost_method.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
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
        """
        Description : Right nodata can not be float or nan with zncc matching_cost_method.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        with pytest.raises((ValueError, DictCheckerError)):
            check_configuration.check_conf(configuration, pandora2d_machine)


class TestDisparityRangeAgainstImageSize:
    """Test that out of image disparity ranges are not allowed."""

    @pytest.fixture()
    def image_path(self, tmp_path):
        path = tmp_path / "tiff_file.tif"
        imsave(path, np.empty((450, 450)))
        return path

    @pytest.fixture()
    def row_disparity(self):
        return {"init": -2, "range": 2}

    @pytest.fixture()
    def col_disparity(self):
        return {"init": -1, "range": 2}

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
            "output": {"path": "path"},
        }

    @pytest.mark.parametrize(
        "row_disparity",
        [
            pytest.param({"init": -456, "range": 5}, id="Out on left"),
            pytest.param({"init": 456, "range": 5}, id="Out on right"),
        ],
    )
    def test_row_disparity_totally_out(self, pandora2d_machine, configuration):
        """
        Description : Totally out disparities should raise an error.
        Data : tmp_path / "tiff_file.tif"
        Requirement : EX_CONF_08
        """
        with pytest.raises(ValueError, match="Row disparity range out of image"):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize(
        "col_disparity",
        [
            pytest.param({"init": -456, "range": 5}, id="Out on top"),
            pytest.param({"init": 456, "range": 5}, id="Out on bottom"),
        ],
    )
    def test_column_disparity_totally_out(self, pandora2d_machine, configuration):
        """
        Description : Totally out disparities should raise an error.
        Data : tmp_path / "tiff_file.tif"
        Requirement : EX_CONF_08
        """
        with pytest.raises(ValueError, match="Column disparity range out of image"):
            check_configuration.check_conf(configuration, pandora2d_machine)

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity"],
        [
            pytest.param({"init": -455, "range": 5}, {"init": 150, "range": 50}, id="Partially out on left"),
            pytest.param({"init": 455, "range": 5}, {"init": 150, "range": 50}, id="Partially out on right"),
            pytest.param({"init": 150, "range": 50}, {"init": -455, "range": 5}, id="Partially out on top"),
            pytest.param({"init": 150, "range": 50}, {"init": 455, "range": 5}, id="Partially out on bottom"),
        ],
    )
    def test_disparity_partially_out(self, pandora2d_machine, configuration):
        """Partially out should not raise error."""
        check_configuration.check_conf(configuration, pandora2d_machine)


class TestCheckDisparity:
    """
    Test check_disparity method
    """

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
                id="Correct disparity with variable initial value",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "second_constant_initial_disparity"},
                id="Correct disparity with constant initial value",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_passes_check_disparity(self, left_img_path, make_input_cfg):
        """
        Test check_disparity method with correct input disparities
        """

        image_metadata = get_metadata(left_img_path)

        check_configuration.check_disparity(image_metadata, make_input_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg", "error_type", "error_message"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "left_img_shape"},
                AttributeError,
                "The disparities in rows and columns must be given as 2 dictionaries",
                id="Col disparity is not a dictionary",
            ),
            pytest.param(
                {"row_disparity": "left_img_shape", "col_disparity": "correct_grid"},
                AttributeError,
                "The disparities in rows and columns must be given as 2 dictionaries",
                id="Row disparity is not a dictionary",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "correct_grid"},
                ValueError,
                "Initial columns and row disparity values must be two strings or two integers",
                id="Initial value is different for columns and rows disparity",
            ),
            pytest.param(
                {"row_disparity": "out_of_image_grid", "col_disparity": "second_correct_grid"},
                ValueError,
                "Row disparity range out of image",
                id="Row disparity grid out of image for one point",
            ),
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "incorrect_disp_dict"},
                ValueError,
                "Column disparity range out of image",
                id="Column disparity dict out of image for one point",
            ),
            pytest.param(
                {"row_disparity": "two_bands_grid", "col_disparity": "correct_grid"},
                AttributeError,
                "Initial disparity grid must be a 1-channel grid",
                id="Row disparity grid has two band",
            ),
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "wrong_size_grid"},
                AttributeError,
                "Initial disparity grids and image must have the same size",
                id="Column disparity grid size is different from image size",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_check_disparity(self, left_img_path, make_input_cfg, error_type, error_message):
        """
        Test check_disparity method with incorrect input disparities
        """

        image_metadata = get_metadata(left_img_path)

        with pytest.raises(error_type, match=error_message):
            check_configuration.check_disparity(image_metadata, make_input_cfg)


@pytest.mark.parametrize(
    "extra_section_name",
    [
        # Let's build a random extra_section_name with a length between 1 and 15 letters
        "".join(random.choices(string.ascii_letters, k=random.randint(1, 15)))
    ],
)
def test_extra_section_is_allowed(correct_input_cfg, correct_pipeline, pandora2d_machine, extra_section_name):
    """
    Description : Should not raise an error if an extra section is added.
    Data :
    - Left image : cones/monoband/left.png
    - Right image : cones/monoband/right.png
    Requirement : EX_CONF_05
    """
    configuration = {**correct_input_cfg, **correct_pipeline, "output": {"path": "here"}, extra_section_name: {}}

    check_configuration.check_conf(configuration, pandora2d_machine)


class TestExpertModeSection:
    """
    Description : Test expert_mode_section.
    """

    def test_expert_mode_section_missing_profile_parameter(self):
        """
        Description : Test if profiling section is missing
        Data :
        Requirement :
        """

        with pytest.raises(MissKeyCheckerError) as exc_info:
            check_configuration.check_expert_mode_section({"expert_mode": {}})
        assert str(exc_info.value) == "Please be sure to set the profiling dictionary"

    @pytest.mark.parametrize(
        ["parameter", "wrong_value_parameter"],
        [
            pytest.param("folder_name", 12, id="error folder name with an int"),
            pytest.param("folder_name", ["folder_name"], id="error folder name with a list"),
            pytest.param("folder_name", {"folder_name": "expert_mode"}, id="error folder name with a dict"),
            pytest.param("folder_name", 12.0, id="error folder name with a float"),
        ],
    )
    def test_configuration_expert_mode(self, parameter, wrong_value_parameter):
        """
        Description : Test if wrong parameters are detected
        Data :
        Requirement :
        """
        with pytest.raises(DictCheckerError) as err:
            check_configuration.check_expert_mode_section({"profiling": {parameter: wrong_value_parameter}})
        assert "folder_name" in err.value.args[0]
