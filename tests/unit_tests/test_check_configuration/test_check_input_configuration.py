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
Test check input step configuration
"""

# pylint: disable=redefined-outer-name,too-many-arguments,too-many-positional-arguments, too-many-lines
import json
import re
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import rasterio
from json_checker import DictCheckerError
from skimage.io import imsave
from pandora.img_tools import rasterio_open, get_metadata

from pandora2d.check_configuration import (
    check_input_section,
    check_disparity,
    check_disparity_ranges_are_inside_image,
    check_disparity_grids,
    load_attributes,
    check_step_from_attributes,
    check_disparity_grids_from_directory_within_image,
    get_dictionary_from_init_grid,
)


class TestCheckInputKey:
    """
    Test on the input key
    Requirement : EX_CONF_01
    """

    def test_check_nominal_case(self, correct_input_cfg) -> None:
        """Test passes when the input section is valid."""
        check_input_section(correct_input_cfg)

    def test_fails_if_input_section_is_missing(self):
        """
        Description : Test raises KeyError if the input section is missing
        Data :
        Requirement : EX_CONF_01
        """
        with pytest.raises(KeyError, match="input key is missing"):
            check_input_section({})


class TestCheckImages:
    """
    Test son left/right images
    Test of  the check_images method from Pandora
    """

    def test_false_input_path_image_should_raise_error(self, false_input_path_image):
        """
        Description : Test raises an error if the image path isn't correct
        Data : cones/monoband/right.png
        Requirement : EX_CONF_09
        """
        with pytest.raises(DictCheckerError, match="img"):
            check_input_section(false_input_path_image)

    def test_fails_with_images_of_different_sizes(self, correct_input_cfg, make_empty_image):
        """
        Description : Test raises an error if left and right images have different sizes
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_11
        """
        correct_input_cfg["input"]["left"]["img"] = str(make_empty_image("left.tiff"))
        correct_input_cfg["input"]["right"]["img"] = str(make_empty_image("right.tiff", shape=(50, 50)))

        with pytest.raises(AttributeError, match="Images must have the same size"):
            check_input_section(correct_input_cfg)


class TestInputSectionWithEstimationStep:
    """
    Test the input section in the configuration file if the estimation step is present
    """

    @pytest.fixture()
    def basic_estimation_cfg(self):
        """Basic estimation configuration"""
        return {"estimation_method": "phase_cross_correlation"}

    def test_check_nominal_case_with_estimation_config(self, correct_input_cfg, basic_estimation_cfg):
        """Test passes with default estimation_config value : basic config."""

        del correct_input_cfg["input"]["col_disparity"]
        del correct_input_cfg["input"]["row_disparity"]
        check_input_section(correct_input_cfg, basic_estimation_cfg)

    def test_estimation_config_with_disparity(self, correct_input_cfg, basic_estimation_cfg):
        """Test raises error with default basic estimation config and disparity in user configuration."""
        with pytest.raises(
            KeyError,
            match="When using estimation, "
            "the col_disparity and row_disparity keys must not be given in the configuration file",
        ):
            check_input_section(correct_input_cfg, basic_estimation_cfg)


class TestNodata:
    """
    Test on the configuration of nodata
    """

    def test_default_nodata(self, correct_input_cfg):
        """
        Description : Sets default nodata to -9999 when not provided
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_04
        """
        del correct_input_cfg["input"]["left"]["nodata"]

        check_input_section(correct_input_cfg)

        assert correct_input_cfg["input"]["left"]["nodata"] == -9999
        assert correct_input_cfg["input"]["right"]["nodata"] == -9999

    @pytest.mark.parametrize(
        ["nodata"],
        [
            pytest.param({"left": -1, "right": -9999}, id="negative value test"),
            pytest.param({"left": 999, "right": -9999}, id="positive & negative value test"),
            pytest.param({"left": -1, "right": "NaN"}, id="NaN and integer test"),
            pytest.param({"left": "NaN", "right": -9999}, id="NaN and integer test (image inversion)"),
            pytest.param({"left": "NaN", "right": "inf"}, id="NaN & inf test"),
        ],
    )
    def test_nodata_value(self, correct_input_cfg, nodata):
        """
        Description : Test passes for valid nodata configurations
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """

        correct_input_cfg["input"]["left"]["nodata"] = nodata["left"]
        correct_input_cfg["input"]["right"]["nodata"] = nodata["right"]

        check_input_section(correct_input_cfg)

    @pytest.mark.parametrize(
        ["nodata"],
        [
            pytest.param({"left": -1.0, "right": -9999}, id="test float value"),
            pytest.param({"left": 999, "right": "-9999"}, id="test string value"),
            pytest.param({"left": [-1], "right": "NaN"}, id="test list value"),
        ],
    )
    def test_fails_with_wrong_nodata_configuration(self, correct_input_cfg, nodata):
        """
        Description : Test raises error for invalid nodata types or values
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """

        correct_input_cfg["input"]["left"]["nodata"] = nodata["left"]
        correct_input_cfg["input"]["right"]["nodata"] = nodata["right"]
        with pytest.raises(DictCheckerError, match="nodata"):
            check_input_section(correct_input_cfg)


class TestMask:
    """
    Test on the mask configuration
    """

    def test_without_mask_on_configuration(self, correct_input_cfg):
        """
        Description : Sets mask to None when not provided
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """
        check_input_section(correct_input_cfg)

        assert correct_input_cfg["input"]["left"]["mask"] is None
        assert correct_input_cfg["input"]["right"]["mask"] is None

    @pytest.mark.parametrize(
        "input_cfg",
        ["correct_input_with_left_mask", "correct_input_with_right_mask", "correct_input_with_left_right_mask"],
    )
    def test_nominal(self, input_cfg, request):
        """
        Description : Test configuration with mask for left and/or right image
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """
        input_cfg = request.getfixturevalue(input_cfg)
        check_input_section(input_cfg)

    @pytest.mark.parametrize("input_cfg", ["false_input_path_left_mask", "false_input_path_right_mask"])
    def test_fails_with_wrong_mask_configuration(self, input_cfg, request):
        """
        Description : Test raises error if a mask path is invalid
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """
        input_cfg = request.getfixturevalue(input_cfg)
        with pytest.raises(DictCheckerError, match="mask"):
            check_input_section(input_cfg)


@pytest.fixture
def user_cfg(make_input_cfg, pipeline_config):
    """
    User configuration
    """
    return {"input": make_input_cfg, **pipeline_config, "output": {"path": "here"}}


@pytest.fixture
def disparity_map_directory(tmp_path):
    """Disparity map directory with files inside."""
    destination = tmp_path / "destination"
    destination.mkdir()
    return destination


@pytest.fixture
def image_shape(left_img_path):
    """image shape"""
    with rasterio.open(left_img_path) as src:
        width = src.width
        height = src.height
    return height, width


@pytest.fixture
def row_disparity_grid(disparity_map_directory, create_disparity_grid_fixture, image_shape):
    """row disparity grid"""
    data = np.ones(image_shape)
    file_path = disparity_map_directory / "row_map.tif"
    # When an absolute path is provided as `suffix_path` to `create_disparity_fixture`, it is used directly
    # instead of being prefixed:
    return create_disparity_grid_fixture(data, 3, file_path)


@pytest.fixture
def row_disparity_directory(tmp_path, row_disparity_grid):
    """row disparity directory"""
    disparity_file_path = Path(row_disparity_grid["init"])
    row_directory = tmp_path / "row_directory"
    row_directory.mkdir()
    disparity_file_path.rename(row_directory / disparity_file_path.name)
    row_disparity_grid.update({"init": str(row_directory)})
    return row_disparity_grid


@pytest.fixture
def col_disparity_grid(disparity_map_directory, create_disparity_grid_fixture, image_shape):
    """col disparity grid"""
    data = np.ones(image_shape)
    file_path = disparity_map_directory / "col_map.tif"
    # When an absolute path is provided as `suffix_path` to `create_disparity_fixture`, it is used directly
    # instead of being prefixed:
    return create_disparity_grid_fixture(data, 3, file_path)


@pytest.fixture
def col_disparity_directory(tmp_path, col_disparity_grid):
    """col disparity directory"""
    disparity_file_path = Path(col_disparity_grid["init"])
    col_directory = tmp_path / "col_directory"
    col_directory.mkdir()
    disparity_file_path.rename(col_directory / disparity_file_path.name)
    col_disparity_grid.update({"init": str(col_directory)})
    return col_disparity_grid


@pytest.fixture
def image_metadata(left_img_path):
    """Load metadata from image"""
    return get_metadata(left_img_path)


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
    def test_passes_check_disparity(self, image_metadata, make_input_cfg, user_cfg):
        """
        Test check_disparity method with correct input disparities
        """
        check_disparity(image_metadata=image_metadata, input_cfg=make_input_cfg, user_cfg=user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "left_img_shape"},
                id="Col disparity is not a dictionary",
            ),
            pytest.param(
                {"row_disparity": "left_img_shape", "col_disparity": "correct_grid"},
                id="Row disparity is not a dictionary",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_when_disparity_is_not_dictionnary(self, image_metadata, make_input_cfg, user_cfg):
        """
        Test check_disparity method with incorrect input disparities : not a dictionnary
        """

        with pytest.raises(AttributeError, match="The disparities in rows and columns must be given as 2 dictionaries"):
            check_disparity(image_metadata=image_metadata, input_cfg=make_input_cfg, user_cfg=user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "constant_initial_disparity", "col_disparity": "correct_grid"},
                id="Initial value is different for columns and rows disparity",
            )
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_when_disparities_are_not_same_type(self, image_metadata, make_input_cfg, user_cfg):
        """
        Test check_disparity method with two disparities different : not same type
        """

        with pytest.raises(
            ValueError, match="Initial columns and row disparity values must be two strings or two integers"
        ):
            check_disparity(image_metadata=image_metadata, input_cfg=make_input_cfg, user_cfg=user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {
                    "row_disparity": "row_disparity_directory",
                    "col_disparity": "col_disparity_grid",
                },
                id="Row: directory; Col: file",
            ),
            pytest.param(
                {
                    "row_disparity": "row_disparity_grid",
                    "col_disparity": "col_disparity_directory",
                },
                id="Row: file; Col: directory",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_when_directory_is_mixed_with_file(self, image_metadata, make_input_cfg, user_cfg):
        """Both disparities must be directories"""
        with pytest.raises(ValueError, match="Directory must not be mixed with file."):
            check_disparity(image_metadata=image_metadata, input_cfg=make_input_cfg, user_cfg=user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {
                    "row_disparity": "row_disparity_directory",
                    "col_disparity": "col_disparity_directory",
                },
                id="Different directories",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_when_directories_are_different(self, image_metadata, make_input_cfg, user_cfg):
        """Both disparities must use the same directory"""
        with pytest.raises(ValueError, match="Row and Col disparities must use the same directory."):
            check_disparity(image_metadata=image_metadata, input_cfg=make_input_cfg, user_cfg=user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                }
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_disparity_directory_is_replaced_by_path_to_file(self, image_metadata, make_input_cfg, user_cfg):
        """When a directory is given, it is replaced by disparity grid paths."""
        input_cfg = deepcopy(make_input_cfg)
        check_disparity(image_metadata=image_metadata, input_cfg=input_cfg, user_cfg=user_cfg)

        assert Path(input_cfg["row_disparity"]["init"]).name == "row_map.tif"
        assert Path(input_cfg["col_disparity"]["init"]).name == "col_map.tif"


class TestCheckDisparityRangesAreInsideImage:
    """Test that out of image disparity ranges are not allowed."""

    @pytest.fixture()
    def image_path(self, tmp_path):
        """Create empty image"""
        path = tmp_path / "tiff_file.tif"
        imsave(path, np.empty((450, 450)))
        return path

    @pytest.fixture()
    def row_disparity(self):
        """Row disparity configuration"""
        return {"init": -2, "range": 2}

    @pytest.fixture()
    def col_disparity(self):
        """Col disparity configuration"""
        return {"init": -1, "range": 2}

    @pytest.mark.parametrize(
        "row_disparity",
        [
            pytest.param({"init": -456, "range": 5}, id="Out on left"),
            pytest.param({"init": 456, "range": 5}, id="Out on right"),
        ],
    )
    def test_row_disparity_totally_out(self, image_path, row_disparity, col_disparity):
        """
        Description : Test raises error if row disparity range is fully outside the image
        Data : tmp_path / "tiff_file.tif"
        Requirement : EX_CONF_08
        """
        image_metadata = get_metadata(image_path)
        with pytest.raises(ValueError, match="Row disparity range out of image"):
            check_disparity_ranges_are_inside_image(image_metadata, row_disparity, col_disparity)

    @pytest.mark.parametrize(
        "col_disparity",
        [
            pytest.param({"init": -456, "range": 5}, id="Out on top"),
            pytest.param({"init": 456, "range": 5}, id="Out on bottom"),
        ],
    )
    def test_column_disparity_totally_out(self, image_path, row_disparity, col_disparity):
        """
        Description : Test raises error if column disparity range is fully outside the image
        Data : tmp_path / "tiff_file.tif"
        Requirement : EX_CONF_08
        """
        image_metadata = get_metadata(image_path)
        with pytest.raises(ValueError, match="Column disparity range out of image"):
            check_disparity_ranges_are_inside_image(image_metadata, row_disparity, col_disparity)

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity"],
        [
            pytest.param({"init": -455, "range": 5}, {"init": 150, "range": 50}, id="Partially out on left"),
            pytest.param({"init": 455, "range": 5}, {"init": 150, "range": 50}, id="Partially out on right"),
            pytest.param({"init": 150, "range": 50}, {"init": -455, "range": 5}, id="Partially out on top"),
            pytest.param({"init": 150, "range": 50}, {"init": 455, "range": 5}, id="Partially out on bottom"),
        ],
    )
    def test_disparity_partially_out(self, image_path, row_disparity, col_disparity):
        """Partially out should not raise error."""
        image_metadata = get_metadata(image_path)
        check_disparity_ranges_are_inside_image(image_metadata, row_disparity, col_disparity)


@pytest.fixture
def disparity_in_dir_readers(make_input_cfg):
    """Row and col DatasetReader"""
    disparity_row_reader = rasterio_open(str(Path(make_input_cfg["row_disparity"]["init"]) / "row_map.tif"))
    disparity_col_reader = rasterio_open(str(Path(make_input_cfg["row_disparity"]["init"]) / "col_map.tif"))
    return (disparity_row_reader, disparity_col_reader)


class TestCheckDisparityGrids:
    """Test check_disparity_grids method"""

    @pytest.fixture
    def row_path(self, make_input_cfg):
        """Path to the row disparity grid"""
        return Path(make_input_cfg["row_disparity"]["init"])

    @pytest.fixture
    def disparity_readers(self, make_input_cfg):
        """Row and col DatasetReader"""
        disparity_row_reader = rasterio_open(make_input_cfg["row_disparity"]["init"])
        disparity_col_reader = rasterio_open(make_input_cfg["col_disparity"]["init"])
        return (disparity_row_reader, disparity_col_reader)

    @pytest.mark.parametrize(
        ["make_input_cfg", "error_message"],
        [
            pytest.param(
                {"row_disparity": "two_bands_grid", "col_disparity": "correct_grid"},
                "Initial disparity grids must be a 1-channel grid",
                id="Row disparity grid has two band",
            ),
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "wrong_size_grid"},
                "Initial disparity grids' sizes do not match",
                id="Column disparity grid size is different from image size",
            ),
            pytest.param(
                {"row_disparity": "wrong_size_grid", "col_disparity": "wrong_size_grid"},
                "Initial disparity grids and image must have the same size",
                id="Disparity grids size are different from image size",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_fails_check_disparity_grids(self, image_metadata, error_message, disparity_readers, row_path, user_cfg):
        """
        Test check_disparity_grids method with incorrect input disparities
        """
        disparity_row_reader, disparity_col_reader = disparity_readers
        with pytest.raises(AttributeError, match=error_message):
            check_disparity_grids(image_metadata, disparity_row_reader, disparity_col_reader, row_path, user_cfg)

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {"row_disparity": "correct_grid", "col_disparity": "correct_grid"},
                id="Row and column disparity grid are two files",
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_passes_check_disparity_grids_with_file(self, image_metadata, disparity_readers, row_path, user_cfg):
        """
        Test check_disparity_grids method with correct input disparities
        """
        disparity_row_reader, disparity_col_reader = disparity_readers
        configuration = deepcopy(user_cfg)
        check_disparity_grids(image_metadata, disparity_row_reader, disparity_col_reader, row_path, configuration)

        assert configuration == user_cfg

    @pytest.mark.parametrize(
        ["make_input_cfg"],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                }
            ),
        ],
        indirect=["make_input_cfg"],
    )
    def test_passes_check_disparity_grids_with_directory(
        self, image_metadata, disparity_in_dir_readers, row_path, user_cfg
    ):
        """
        Test check_disparity_grids method with correct input disparities
        """
        disparity_row_reader, disparity_col_reader = disparity_in_dir_readers
        configuration = deepcopy(user_cfg)
        check_disparity_grids(image_metadata, disparity_row_reader, disparity_col_reader, row_path, configuration)

        assert "attributes" in configuration

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        ["correct_grid_shape", "second_correct_grid_shape", "roi_user", "roi_gt"],
        [
            pytest.param(
                (7, 7),
                (7, 7),
                {"row": {"first": 0, "last": 18}, "col": {"first": 2, "last": 24}},
                {"row": {"first": 0, "last": 6}, "col": {"first": 0, "last": 6}},
            ),
        ],
    )
    def test_check_conf_with_roi_grid_and_user_roi(
        self,
        image_metadata,
        disparity_in_dir_readers,
        row_path,
        user_cfg,
        correct_grid_shape,
        second_correct_grid_shape,
        roi_user,
        roi_gt,
        caplog,
    ):
        """
        Check that the ROI constructed from the disparity grid overwrites the user ROI
        """

        disparity_row_reader, disparity_col_reader = disparity_in_dir_readers
        configuration = deepcopy(user_cfg)
        configuration["ROI"] = roi_user

        with caplog.at_level(logging.WARNING):
            check_disparity_grids(image_metadata, disparity_row_reader, disparity_col_reader, row_path, configuration)

        assert "attributes" in configuration
        assert configuration["ROI"] == roi_gt
        assert (
            "The ROI given in the user configuration will be replaced by the ROI derived from the disparity grids."
            in caplog.messages
        )


@pytest.fixture
def step():
    """Step in attributes file"""
    return [1, 1]


@pytest.fixture
def step_offset():
    """Offset to add to step in attributes file."""
    return [0, 0]


@pytest.fixture
def attributes_file(tmp_path, step, step_offset, origin_coordinates):
    """Create attributes files"""
    file_path = tmp_path / "attributes.json"
    with open(file_path, "w", encoding="utf-8") as fd:
        json.dump(
            {
                "step": {"row": step[0] + step_offset[0], "col": step[1] + step_offset[1]},
                "origin_coordinates": {"row": origin_coordinates["row"], "col": origin_coordinates["col"]},
            },
            fd,
        )
    return file_path


class TestCheckLoadAttributes:
    """Test load_attributes method"""

    def test_fails_when_attributes_file_is_missing(self, tmp_path, attributes_file):
        """The directory must contain attribute file."""
        attributes_file.unlink()
        with pytest.raises(FileNotFoundError, match=re.escape(str(attributes_file))):
            load_attributes(tmp_path)

    def test_passes(self, tmp_path, attributes_file):
        """The directory contains attribute file."""
        attributes = load_attributes(tmp_path)
        expected_attributes = json.loads(attributes_file.read_text(encoding="utf-8"))

        assert expected_attributes == attributes


@pytest.fixture
def attributes(attributes_file):
    return load_attributes(attributes_file.parent)


class TestCheckStepFromAttributes:
    """Test check_step_from_attributes method"""

    @pytest.mark.parametrize(
        ["step", "step_offset", "expected_value"],
        [
            pytest.param([7, 7], [7, 7], [1, 1]),
        ],
    )
    def test_check_conf_fails_when_steps_differs(self, attributes, step, step_offset, expected_value):
        """Check conf should fail when step from pipeline config and step from attributes differs."""
        attributes_step = [attributes["step"]["row"], attributes["step"]["col"]]
        message = f"Initial disparity grid step {attributes_step} does not match configuration step {expected_value}."
        with pytest.raises(AttributeError, match=re.escape(message)):
            check_step_from_attributes(attributes, expected_value)

    @pytest.mark.parametrize(
        ["expected_value"],
        [
            pytest.param([1, 1]),
        ],
    )
    def test_check_conf_pass_when_steps_equals(self, attributes, expected_value):
        """Check conf should fail when step from pipeline config and step from attributes equals."""
        check_step_from_attributes(attributes, expected_value)


class TestCheckDisparityGridsFromDirectoryWithinImage:
    """Test check_disparity_grids_from_directory_within_image method"""

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "step",
        ],
        [
            pytest.param((10, 10), (10, 10), {"row": 0, "col": 0}, [200, 200], id="Step"),
            pytest.param((10, 10), (10, 10), {"row": 370, "col": 445}, [1, 1], id="Origin"),
        ],
    )
    def test_fails_when_disparity_grids_bounds_are_out_of_image(
        self, attributes, disparity_in_dir_readers, image_metadata, correct_grid_shape, second_correct_grid_shape
    ):
        """The disparity grids must remain inside the image boundaries after expansion by step and/or origin."""
        message = "Initial disparity grid is not inside image boundaries."
        with pytest.raises(AttributeError, match=re.escape(message)):
            check_disparity_grids_from_directory_within_image(attributes, disparity_in_dir_readers[0], image_metadata)

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize(
        [
            "correct_grid_shape",
            "second_correct_grid_shape",
            "origin_coordinates",
            "expected",
        ],
        [
            pytest.param((375, 450), (375, 450), {"row": 0, "col": 0}, None, id="No ROI"),
            pytest.param(
                (10, 10),
                (10, 10),
                {"row": 0, "col": 0},
                {"row": {"first": 0, "last": 9}, "col": {"first": 0, "last": 9}},
                id="ROI without special origin",
            ),
            pytest.param(
                (5, 9),
                (5, 9),
                {"row": 13, "col": 47},
                {"row": {"first": 13, "last": 17}, "col": {"first": 47, "last": 55}},
                id="ROI with special origin",
            ),
        ],
    )
    def test_passes(
        self,
        attributes,
        disparity_in_dir_readers,
        image_metadata,
        correct_grid_shape,
        second_correct_grid_shape,
        expected,
    ):
        """The disparity grids must remain inside the image boundaries after expansion by step and/or origin."""
        roi = check_disparity_grids_from_directory_within_image(attributes, disparity_in_dir_readers[0], image_metadata)

        assert roi == expected


class TestGetDictionaryFromInitGrid:
    """
    Check get_dictionary_from_init_grid method
    """

    @pytest.fixture()
    def correct_grid_data(self, no_data_disp):
        """
        correct_grid_data fixture for get_dictionary_from_init_grid method
        """

        data = np.array(
            [
                [5.0, 1.0, 2.0, 0.0, 3.0],
                [1.0, 2.0, 2.0, 2.0, 1.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [3.0, 3.0, 1.0, 0.0, 5.0],
                [2.0, 1.0, 0.0, 5.0, 4.0],
            ]
        )

        if no_data_disp is not None:
            data[0, 0] = no_data_disp
            data[1, 2] = no_data_disp
            data[3, 1] = no_data_disp

        return data

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize("correct_grid_shape", [(5, 5)])
    @pytest.mark.parametrize(
        [
            "no_data_disp",
            "gt_disparity_dict",
        ],
        [
            pytest.param(None, {"init": 5, "range": 5}, id="No data disparity is None"),
            pytest.param(5, {"init": 4, "range": 5}, id="Integer no data disparity"),
            pytest.param(-5, {"init": 5, "range": 5}, id="Negative integer no data disparity"),
            pytest.param(np.nan, {"init": 5, "range": 5}, id="NaN no data disparity"),
            pytest.param(np.inf, {"init": 5, "range": 5}, id="Inf no data disparity"),
            pytest.param(-np.inf, {"init": 5, "range": 5}, id="-Inf no data disparity"),
        ],
    )
    def test_get_dictionary_from_init_grid(
        self,
        disparity_in_dir_readers,
        correct_grid_shape,
        correct_grid_data,
        no_data_disp,
        gt_disparity_dict,
    ):
        """
        Check get_dictionary_from_init_grid method
        """

        # 5 is the range value used for the same_sized_grid_directory fixture
        disparity_dict = get_dictionary_from_init_grid(disparity_in_dir_readers[0], 5)

        assert disparity_dict == gt_disparity_dict

    @pytest.mark.parametrize(
        [
            "make_input_cfg",
        ],
        [
            pytest.param(
                {
                    "row_disparity": "same_sized_grid_directory",
                    "col_disparity": "same_sized_grid_directory",
                },
            )
        ],
        indirect=["make_input_cfg"],
    )
    @pytest.mark.parametrize("correct_grid_shape", [(5, 5)])
    @pytest.mark.parametrize(
        [
            "correct_grid_data",
            "no_data_disp",
        ],
        [
            pytest.param(np.full((5, 5), 5.0), 5, id="Integer no data disparity"),
            pytest.param(np.full((5, 5), np.nan), np.nan, id="NaN no data disparity"),
            pytest.param(np.full((5, 5), np.inf), np.inf, id="Inf no data disparity"),
            pytest.param(np.full((5, 5), -np.inf), -np.inf, id="-Inf no data disparity"),
        ],
    )
    def test_fails_when_init_disp_full_of_no_data_values(
        self, disparity_in_dir_readers, correct_grid_shape, correct_grid_data, no_data_disp
    ):
        """
        Check get_dictionary_from_init_grid method fails when initial disparity grid is
        full of no data disparity values.
        """
        # 5 is the range value used for the same_sized_grid_directory fixture
        with pytest.raises(ValueError, match="Initial disparity grid is full of invalid values"):
            get_dictionary_from_init_grid(disparity_in_dir_readers[0], 5)
