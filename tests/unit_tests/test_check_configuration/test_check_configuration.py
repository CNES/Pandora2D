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
Test method on check_configuration file
"""

import random
import string

import numpy as np
import pytest
import xarray as xr
from json_checker import DictCheckerError

from pandora2d.check_configuration import check_datasets, get_section_config, check_right_nodata_condition, check_conf
from pandora2d.img_tools import add_disparity_grid, create_datasets_from_inputs


class TestCheckDatasets:
    """Tests for check_datasets"""

    @pytest.fixture()
    def datasets(self, left_img_path, right_img_path):
        """Build valid left and right datasets for tests"""
        input_cfg = {
            "left": {"img": left_img_path, "nodata": -9999},
            "right": {"img": right_img_path, "nodata": -9999},
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 3},
        }
        return create_datasets_from_inputs(input_cfg)

    def test_nominal(self, datasets):
        """
        Nominal case with valid image datasets
        """
        dataset_left, dataset_right = datasets
        check_datasets(dataset_left, dataset_right)

    def test_fails_with_wrong_dimension(self):
        """
        Description : Test fails when left and right image shapes differ.
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
            check_datasets(dataset_left, dataset_right)
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
        Description : Test fails when required disparity variables are missing
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
            check_datasets(dataset_left, dataset_right)
        assert str(exc_info.value) == "left dataset must have column and row disparities DataArrays"


class TestGetSectionConfig:
    """Test for get_section_config"""

    @pytest.fixture
    def null_config(self):
        return {}

    @pytest.mark.parametrize(
        ["key", "expected"],
        [
            pytest.param("ROI", "correct_roi_sensor", id="Get roi section"),
            pytest.param("toto", "null_config", id="No key in configuration"),
        ],
    )
    def test_get_section_config(self, correct_roi_sensor, key, expected, request):
        """Retrieve an existing section or return an empty dictionary"""
        assert get_section_config(correct_roi_sensor, key) == request.getfixturevalue(expected)


class TestCheckConfMatchingCostNodataCondition:
    """Test for check_right_nodata_condition"""

    @pytest.fixture()
    def input_configuration(self, right_nodata, left_img_path, right_img_path):
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
            }
        }

    @pytest.fixture
    def pipeline_configuration(self, matching_cost_method):
        return {
            "pipeline": {
                "matching_cost": {"matching_cost_method": matching_cost_method, "window_size": 1},
            }
        }

    @pytest.mark.parametrize("right_nodata", ["NaN", 0.1, "inf", None])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd"])
    def test_sad_or_ssd_fails_with(self, input_configuration, pipeline_configuration):
        """
        Description : Right nodata must be an integer with sad or ssd matching_cost_method.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        Requirement : EX_CONF_08
        """
        with pytest.raises((ValueError, DictCheckerError)):
            check_right_nodata_condition(input_configuration["input"], pipeline_configuration["pipeline"])

    @pytest.mark.parametrize("right_nodata", [432])
    @pytest.mark.parametrize("matching_cost_method", ["sad", "ssd", "zncc"])
    def test_passes_with_int(self, input_configuration, pipeline_configuration):
        """Right nodata must be an integer."""
        check_right_nodata_condition(input_configuration["input"], pipeline_configuration["pipeline"])

    @pytest.mark.parametrize("right_nodata", ["NaN", "inf"])
    @pytest.mark.parametrize("matching_cost_method", ["zncc"])
    def test_zncc_passes_with(self, input_configuration, pipeline_configuration):
        """Right nodata can be inf or nan with zncc matching_cost_method."""
        check_right_nodata_condition(input_configuration["input"], pipeline_configuration["pipeline"])


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

    check_conf(configuration, pandora2d_machine)
