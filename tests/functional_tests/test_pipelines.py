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

"""
Run pandora2d configurations from end to end.
"""

# pylint: disable=redefined-outer-name

import json
from copy import deepcopy
from typing import Dict

import pytest

import numpy as np
import rasterio


def remove_extra_keys(extended: dict, reference: dict):
    """
    Removes the extra keys in the `extended` dictionary that are not present in the `reference` dictionary.

    :param extended: The dictionary that may contain extra keys.
    :type extended: dict
    :param reference: The reference dictionary that contains the desired keys.
    :type reference: dict
    :return: A copy of the `extended` dictionary with only the keys present in the `reference` dictionary.
    :rtype: dict

    :Example:

    >>> extended = {"a": 1, "b": 2, "c": 3}
    >>> reference = {"a": 1, "b": 2}
    >>> remove_extra_keys(extended, reference)
    {'a': 1, 'b': 2}
    """
    extended_copy = deepcopy(extended)
    keys_only_in_extended = extended_copy.keys() - reference.keys()
    for key in keys_only_in_extended:
        extended_copy.pop(key)
    for extended_key, extended_value in extended_copy.items():
        reference_value = reference[extended_key]
        if isinstance(extended_value, dict) and isinstance(reference_value, dict):
            extended_copy[extended_key] = remove_extra_keys(extended_value, reference_value)
    return extended_copy


class TestRemoveExtrakeys:
    """Various tests on remove_extra_keys function."""

    def test_is_subset_dict(self):
        """Dicts have common keys."""
        reference = {"a": 2, "c": 3}
        value = {"a": 2, "b": {"aa": 4}, "c": 3}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_is_subset_dict_but_not_same_order(self):
        """Order of keys should not have influence."""
        reference = {"a": 2, "b": {"aa": 4}}
        value = {"b": {"aa": 4}, "a": 2}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_nested_subsets(self):
        """Values are dict that and result should recursively equal to reference."""
        reference = {"b": {"aa": 4}, "c": {"cc": 3}}
        value = {"a": 2, "b": {"aa": 4}, "c": {"cc": 3}}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_no_common_key(self):
        """No common key means all keys are extra keys to be removed."""
        reference = {"c": 2}
        value = {"a": 2, "b": {"aa": 4}}

        result = remove_extra_keys(value, reference)

        assert result == {}


@pytest.mark.parametrize(
    "roi",
    [
        pytest.param({}, id="No ROI"),
        pytest.param(
            {
                "ROI": {
                    "col": {"first": 3, "last": 7},
                    "row": {"first": 5, "last": 8},
                }
            },
            id="With ROI",
        ),
    ],
)
def test_monoband_with_nodata_not_nan(
    run_pipeline, correct_input_cfg, correct_pipeline_without_refinement, roi, tmp_path
):
    """
    Description : Test a configuration with monoband images.
    Data :
    - Left image : cones/monoband/left.png
    - Right image : cones/monoband/right.png
    Requirement : EX_CONF_00, EX_CONF_06
    """
    configuration = {
        **correct_input_cfg,
        **correct_pipeline_without_refinement,
        **roi,
        **{"output": {"path": "relative"}},
    }
    configuration["input"]["left"]["nodata"] = -9999

    run_pipeline(configuration)

    with open(tmp_path / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"

    # Test for report
    with open(tmp_path / "relative" / "disparity_map" / "report.json", encoding="utf8") as report_file:
        report = json.load(report_file)

    assert report["statistics"]["disparity"].keys() == {"row", "col"}


@pytest.mark.xfail(reason="saved nan in nodata is not valid json and is not comparable to nan")
def test_monoband_with_nan_nodata(run_pipeline, correct_input_cfg, correct_pipeline_without_refinement, tmp_path):
    """
    Description : Test a configuration with monoband images and left nodata set to NaN.
    Data :
    - Left image : cones/monoband/left.png
    - Right image : cones/monoband/right.png
    Requirement : EX_CONF_00, EX_CONF_06
    """
    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement, **{"output": {"path": str(tmp_path)}}}

    run_pipeline(configuration)

    with open(tmp_path / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"


@pytest.mark.xfail(reason="Multiband is not managed")
def test_multiband(run_pipeline, correct_multiband_input_cfg, correct_pipeline_without_refinement, tmp_path):
    """
    Description : Test a configuration with multiband images.
    Data :
    - Left image : cones/multibands/left.tif
    - Right image : cones/multibands/right.tif
    Requirement : EX_CONF_00, EX_CONF_06, EX_CONF_12
    """
    configuration: Dict[str, Dict] = {
        **correct_multiband_input_cfg,
        **correct_pipeline_without_refinement,
        **{"output": {"path": str(tmp_path)}},
    }

    run_pipeline(configuration)

    with open(tmp_path / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"


def test_optical_flow_configuration(run_pipeline, correct_input_cfg, correct_pipeline_with_optical_flow, tmp_path):
    """
    Description : Test optical_flow configuration has a window_size and a step identical to matching_cost step.
    Data :
    - Left image : cones/monoband/left.png
    - Right image : cones/monoband/right.png
    Requirement : EX_CONF_00, EX_CONF_06
    """
    configuration: Dict[str, Dict] = {
        **correct_input_cfg,
        **correct_pipeline_with_optical_flow,
        **{"output": {"path": str(tmp_path)}},
    }
    configuration["pipeline"]["refinement"]["iterations"] = 1

    run_pipeline(configuration)

    with open(tmp_path / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    matching_cost_cfg = output_config["pipeline"]["matching_cost"]
    refinement_cfg = output_config["pipeline"]["refinement"]

    # Check window_size and step parameters
    assert matching_cost_cfg["window_size"] == refinement_cfg["window_size"]
    assert matching_cost_cfg["step"] == refinement_cfg["step"]


@pytest.mark.parametrize("input_cfg", ["correct_input_with_left_mask", "correct_input_with_right_mask"])
def test_configuration_with_mask(run_pipeline, input_cfg, correct_pipeline_without_refinement, request, tmp_path):
    """
    Description : Test mask configuration
    """
    input_cfg = request.getfixturevalue(input_cfg)

    configuration = {**input_cfg, **correct_pipeline_without_refinement, **{"output": {"path": str(tmp_path)}}}

    run_pipeline(configuration)

    with open(tmp_path / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"

    # Test for report
    with open(tmp_path / "disparity_map" / "report.json", encoding="utf8") as report_file:
        report = json.load(report_file)

    assert report["statistics"]["disparity"].keys() == {"row", "col"}


@pytest.mark.parametrize(
    ["make_input_cfg", "pipeline"],
    [
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_without_refinement",
            id="Pipeline with disparity grids",
        ),
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_with_dichotomy_python",
            id="Pipeline with disparity grids and dichotomy python",
        ),
        pytest.param(
            {"row_disparity": "correct_grid", "col_disparity": "second_correct_grid"},
            "correct_pipeline_with_dichotomy_cpp",
            id="Pipeline with disparity grids and dichotomy cpp",
        ),
    ],
    indirect=["make_input_cfg"],
)
def test_disparity_grids(run_pipeline, make_input_cfg, pipeline, request, tmp_path):
    """
    Description: Test pipeline with disparity grids
    """

    configuration = {
        "input": make_input_cfg,
        "ROI": {"col": {"first": 210, "last": 240}, "row": {"first": 210, "last": 240}},
        **request.getfixturevalue(pipeline),
        **{"output": {"path": str(tmp_path)}},
    }
    configuration["pipeline"]["disparity"]["invalid_disparity"] = np.nan

    run_pipeline(configuration)

    with rasterio.open(tmp_path / "disparity_map" / "row_map.tif") as src:
        row_map = src.read(1)
    with rasterio.open(tmp_path / "disparity_map" / "col_map.tif") as src:
        col_map = src.read(1)

    non_nan_row_map = ~np.isnan(row_map)
    non_nan_col_map = ~np.isnan(col_map)

    # Minimal and maximal disparities corresponding to correct_grid_path fixture
    min_max_disp_row = np.array(
        [
            np.tile([[-3], [-5], [-2]], (375 // 3 + 1, 450))[210:241, 210:241],
            np.tile([[7], [5], [8]], (375 // 3 + 1, 450))[210:241, 210:241],
        ]
    )

    # Minimal and maximal disparities corresponding to second_correct_grid_path fixture
    min_max_disp_col = np.array(
        [
            np.tile([[0, -26, -6]], (375, 450 // 3 + 1))[210:241, 210:241],
            np.tile([[10, -16, 4]], (375, 450 // 3 + 1))[210:241, 210:241],
        ]
    )

    # Checks that the resulting disparities are well within the ranges created from the input disparity grids
    assert np.all(
        (row_map[non_nan_row_map] >= min_max_disp_row[0, ::][non_nan_row_map])
        & (row_map[non_nan_row_map] <= min_max_disp_row[1, ::][non_nan_row_map])
    )
    assert np.all(
        (col_map[non_nan_col_map] >= min_max_disp_col[0, ::][non_nan_col_map])
        & (col_map[non_nan_col_map] <= min_max_disp_col[1, ::][non_nan_col_map])
    )


@pytest.mark.usefixtures("reset_profiling")
@pytest.mark.parametrize(
    ["ground_truth", "configuration_expert", "file_exists"],
    [
        pytest.param(
            [".csv", ".pdf"],
            {"expert_mode": {"profiling": {"folder_name": "expert_mode"}}},
            True,
            id="Expert mode",
        ),
        pytest.param([], {}, False, id="No expert mode"),
    ],
)
def test_expert_mode(
    ground_truth,
    configuration_expert,
    run_pipeline,
    file_exists,
    correct_input_cfg,
    correct_pipeline_without_refinement,
    tmp_path,
):
    """
    Description : Test default expert mode outputs
    Data :
    - Left image : cones/monoband/left.png
    - Right image : cones/monoband/right.png
    """

    configuration = {
        **correct_input_cfg,
        **correct_pipeline_without_refinement,
        **configuration_expert,
        **{"output": {"path": str(tmp_path)}},
    }

    run_pipeline(configuration)

    output_expert_dir = tmp_path / "expert_mode"

    assert output_expert_dir.exists() == file_exists

    if output_expert_dir.exists():
        file_extensions = [f.suffix for f in output_expert_dir.iterdir() if f.is_file()]
        assert set(file_extensions) == set(ground_truth)


class TestAttributes:
    """
    Test that attributes are correctly saved in output directory
    """

    @pytest.mark.parametrize(
        ["step"],
        [
            pytest.param(
                [1, 1],
                id="without ROI and step=[1,1]",
            ),
            pytest.param(
                [3, 2],
                id="without ROI and step=[3,2]",
            ),
        ],
    )
    def test_attributes(self, run_pipeline, configuration, step, tmp_path):
        """
        Description : Test saved attributes without ROI.
        """

        run_pipeline(configuration)

        with rasterio.open(configuration["input"]["left"]["img"]) as src:
            left_crs = src.crs
            left_transform = src.transform

        # Test for attributes
        with open(tmp_path / "disparity_map" / "attributes.json", encoding="utf8") as attrs_file:
            attrs = json.load(attrs_file)
            attrs["transform"] = rasterio.Affine(*attrs["transform"])

        assert attrs["offset"]["row"] == 0
        assert attrs["offset"]["col"] == 0
        assert attrs["step"]["row"] == step[0]
        assert attrs["step"]["col"] == step[1]
        assert attrs["crs"] == left_crs
        # Apply the same transformation as the one done in common.adjust_georeferencement because we have a step
        assert attrs["transform"] == left_transform * rasterio.Affine.scale(step[1], step[0])
        assert attrs["invalid_disp"] == configuration["pipeline"]["disparity"]["invalid_disparity"]

    @pytest.mark.parametrize(
        ["roi", "step"],
        [
            pytest.param(
                {
                    "col": {"first": 3, "last": 7},
                    "row": {"first": 5, "last": 8},
                },
                [1, 1],
                id="with ROI and step=[1,1]",
            ),
            pytest.param(
                {
                    "col": {"first": 3, "last": 7},
                    "row": {"first": 5, "last": 8},
                },
                [3, 2],
                id="with ROI and step=[3,2]",
            ),
        ],
    )
    def test_attributes_with_roi(self, run_pipeline, configuration, roi, step, tmp_path):
        """
        Description : Test saved attributes with ROI.
        """

        configuration["ROI"] = roi

        run_pipeline(configuration)

        with rasterio.open(configuration["input"]["left"]["img"]) as src:
            left_crs = src.crs
            left_transform = src.transform

        # Test for attributes
        with open(tmp_path / "disparity_map" / "attributes.json", encoding="utf8") as attrs_file:
            attrs = json.load(attrs_file)
            attrs["transform"] = rasterio.Affine(*attrs["transform"])

        assert attrs["offset"]["row"] == roi["row"]["first"]
        assert attrs["offset"]["col"] == roi["col"]["first"]
        assert attrs["step"]["row"] == step[0]
        assert attrs["step"]["col"] == step[1]
        assert attrs["crs"] == left_crs
        # Apply the same transformation as the one done in common.adjust_georeferencement because we have a ROI and step
        assert attrs["transform"] == left_transform * rasterio.Affine.translation(
            roi["col"]["first"], roi["row"]["first"]
        ) * rasterio.Affine.scale(step[1], step[0])
        assert attrs["invalid_disp"] == configuration["pipeline"]["disparity"]["invalid_disparity"]

    def test_attributes_without_step(
        self, run_pipeline, correct_input_cfg, correct_pipeline_without_refinement, tmp_path
    ):
        """
        Description : Test saved attributes when step is not given in user cfg.
        Data :
        - Left image : cones/monoband/left.png
        - Right image : cones/monoband/right.png
        """

        # The configuration used in this test is different from the one used in the other tests of the class,
        # because here we want to test a configuration where the step value is not specified.
        configuration = {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            **{"output": {"path": str(tmp_path)}},
        }

        run_pipeline(configuration)

        # Test for attributes
        with open(tmp_path / "disparity_map" / "attributes.json", encoding="utf8") as attrs_file:
            attrs = json.load(attrs_file)

        assert attrs["offset"]["row"] == 0
        assert attrs["offset"]["col"] == 0
        assert attrs["step"]["row"] == 1
        assert attrs["step"]["col"] == 1
        assert attrs["crs"] is None
        assert attrs["transform"] is None
        assert attrs["invalid_disp"] == configuration["pipeline"]["disparity"]["invalid_disparity"]
