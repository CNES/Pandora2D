# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from json_checker.core.exceptions import MissKeyCheckerError

import pandora2d
from pandora2d import Pandora2DMachine
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs


def remove_extra_keys(extended: dict, reference: dict) -> dict:
    """
    Removes the extra keys in the `extended` dictionary that are not present in the `reference` dictionary.

    :param extended: The dictionary that may contain extra keys.
    :param reference: The reference dictionary that contains the desired keys.
    :return: A copy of the `extended` dictionary with only the keys present in the `reference` dictionary.

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


def transform_config_to_cones(config: dict, support_files: dict) -> dict:
    """
    Transform a configuration to use cones/monoband images from support_files.

    Replaces maricopa image, mask and disparity grid paths with absolute paths
    provided in support_files. Also adjusts the nodata value for cones (255 vs -9999).

    :param config: Raw configuration dictionary (with maricopa paths).
    :param support_files: Dict mapping file roles to absolute path strings.
        Expected keys: 'left', 'right', 'left_mask', 'right_mask',
        'init_col_disparity_grid', 'init_row_disparity_grid'.
    :return: A copy of the configuration with cones absolute paths.
    """
    transformed = deepcopy(config)

    transformed["input"]["left"]["img"] = support_files["left"]
    transformed["input"]["right"]["img"] = support_files["right"]

    if transformed["input"]["left"].get("mask"):
        transformed["input"]["left"]["mask"] = support_files["left_mask"]
    if transformed["input"]["right"].get("mask"):
        transformed["input"]["right"]["mask"] = support_files["right_mask"]

    if "col_disparity" in transformed["input"]:
        if isinstance(transformed["input"]["col_disparity"]["init"], str):
            transformed["input"]["col_disparity"]["init"] = support_files["init_col_disparity_grid"]
    if "row_disparity" in transformed["input"]:
        if isinstance(transformed["input"]["row_disparity"]["init"], str):
            transformed["input"]["row_disparity"]["init"] = support_files["init_row_disparity_grid"]

    # Update nodata value for cones (255 instead of -9999)
    if "nodata" in transformed["input"]["left"]:
        transformed["input"]["left"]["nodata"] = 255
    if "nodata" in transformed["input"]["right"]:
        transformed["input"]["right"]["nodata"] = 255

    return transformed


DATA_SAMPLES_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data_samples" / "json_conf_files"

# Configs excluded from reusability tests because they are too slow to run in CI.
_SKIPPED_DATA_SAMPLES = {"a_dichotomy_python_pipeline"}


def filelist_parametrize_generator():
    """
    Generate parametrized test data from sample JSON configurations.

    Loads each raw JSON config and yields it with its source file path.
    Path substitution (maricopa → cones) is deferred to test time via the
    cones_support_files fixture so that support files are created in tmp_path.
    Configs listed in _SKIPPED_DATA_SAMPLES are excluded.
    """
    for config_file in sorted(DATA_SAMPLES_CONFIG_DIR.glob("*.json")):
        if config_file.suffix != ".json" or config_file.stem.endswith("_output"):
            continue
        if config_file.stem in _SKIPPED_DATA_SAMPLES:
            continue

        with config_file.open(encoding="utf8") as sample_file:
            configuration = json.load(sample_file)

        yield pytest.param((configuration, config_file), id=config_file.stem)


def is_estimation_pipeline(config_file: Path) -> bool:
    """
    Check if a configuration file uses the estimation pipeline.

    Estimation pipelines have known limitations with reentrance:
    the output config contains estimated_shifts which causes validation errors
    on second run (unless check_configuration.py is modified to support it).

    :param config_file: Path to the configuration file
    :return: True if the config uses estimation, False otherwise
    """
    return "estimation" in config_file.stem


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
def test_monoband_with_nodata_not_nan(run_pipeline, correct_input_cfg, correct_pipeline_without_refinement, roi):
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

    input_config_dir = run_pipeline(configuration)

    with open(input_config_dir / "relative" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)
    result["output"]["path"] = str(Path(result["output"]["path"]).relative_to(input_config_dir))

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"

    # Test for report
    with open(input_config_dir / "relative" / "disparity_map" / "report.json", encoding="utf8") as report_file:
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
    configuration = {
        **correct_input_cfg,
        **correct_pipeline_without_refinement,
        **{"output": {"path": str(tmp_path / "output")}},
    }

    run_pipeline(configuration)

    with open(tmp_path / "output" / "config.json", encoding="utf8") as output_file:
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
    configuration: dict[str, dict] = {
        **correct_multiband_input_cfg,
        **correct_pipeline_without_refinement,
        **{"output": {"path": str(tmp_path / "output")}},
    }

    run_pipeline(configuration)

    input_config_dir = run_pipeline(configuration)

    with open(input_config_dir / "output" / "config.json", encoding="utf8") as output_file:
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
    configuration: dict[str, dict] = {
        **correct_input_cfg,
        **correct_pipeline_with_optical_flow,
        **{"output": {"path": str(tmp_path / "output")}},
    }
    configuration["pipeline"]["refinement"]["iterations"] = 1

    run_pipeline(configuration)

    with open(tmp_path / "output" / "config.json", encoding="utf8") as output_file:
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

    configuration = {
        **input_cfg,
        **correct_pipeline_without_refinement,
        **{"output": {"path": str(tmp_path / "output")}},
    }

    run_pipeline(configuration)

    with open(tmp_path / "output" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"

    # Test for report
    with open(tmp_path / "output" / "disparity_map" / "report.json", encoding="utf8") as report_file:
        report = json.load(report_file)

    assert report["statistics"]["disparity"].keys() == {"row", "col"}


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

        assert attrs["origin_coordinates"]["row"] == 0
        assert attrs["origin_coordinates"]["col"] == 0
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

        assert attrs["origin_coordinates"]["row"] == roi["row"]["first"]
        assert attrs["origin_coordinates"]["col"] == roi["col"]["first"]
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

        assert attrs["origin_coordinates"]["row"] == 0
        assert attrs["origin_coordinates"]["col"] == 0
        assert attrs["step"]["row"] == 1
        assert attrs["step"]["col"] == 1
        assert attrs["crs"] is None
        assert attrs["transform"] is None
        assert attrs["invalid_disp"] == configuration["pipeline"]["disparity"]["invalid_disparity"]


class TestEstimation:
    """
    Check that pipeline with estimation step is correctly executed
    """

    @pytest.fixture()
    def input_for_estimation(self, correct_input_cfg):
        """
        Input for estimation pipeline without disparity
        """
        del correct_input_cfg["input"]["col_disparity"]
        del correct_input_cfg["input"]["row_disparity"]
        return correct_input_cfg

    @pytest.fixture()
    def range_row(self):
        """
        Range row for estimation
        """
        return 5

    @pytest.fixture()
    def range_col(self):
        """
        Range col for estimation
        """
        return 5

    @pytest.fixture()
    def estimation_pipeline(self, range_row, range_col, correct_pipeline_without_refinement):
        """
        Pipeline with estimation only
        """
        return {
            "pipeline": {
                "estimation": {
                    "estimation_method": "phase_cross_correlation",
                    "range_row": range_row,
                    "range_col": range_col,
                    "sample_factor": 100,
                },
                **correct_pipeline_without_refinement["pipeline"],
            }
        }

    @pytest.fixture()
    def estimation_cfg(self, input_for_estimation, estimation_pipeline, tmp_path):
        """
        Estimation configuration
        """

        return {**input_for_estimation, **estimation_pipeline, **{"output": {"path": str(tmp_path)}}}

    def test_run_estimation_pipeline(self, estimation_cfg, run_pipeline, tmp_path):
        """
        Description: Test pipeline with estimation
        """

        run_pipeline(estimation_cfg)

        with open(tmp_path / "config.json", encoding="utf8") as output_file:
            output_config = json.load(output_file)

        # Check output configuration information about estimation
        estimation_cfg = output_config["pipeline"]["estimation"]
        assert "estimated_shifts" in estimation_cfg
        assert "error" in estimation_cfg
        assert "phase_diff" in estimation_cfg

    def test_raise_error_when_disp_given_in_cfg(self, estimation_cfg, run_pipeline):
        """
        Description: Test that a pipeline with the estimation and disparities in the input cfg raises an error
        """

        estimation_cfg["input"]["row_disparity"] = {"init": 0, "range": 2}
        estimation_cfg["input"]["col_disparity"] = {"init": -1, "range": 3}
        with pytest.raises(
            KeyError,
            match="When using estimation, "
            "the col_disparity and row_disparity keys must not be given in the configuration file",
        ):
            run_pipeline(estimation_cfg)

    @pytest.mark.parametrize(
        [
            "roi",
            "range_row",
            "range_col",
            "window_size",
            "estimated_d_row",
            "estimated_d_col",
            "expected_row",
            "expected_col",
        ],
        [
            pytest.param(
                {
                    "row": {"first": 50, "last": 75},
                    "col": {"first": 50, "last": 60},
                },
                5,
                5,
                5,
                [0, 10],
                [-5, 5],
                # Estimated row disparity is [0, 10]
                # So up_margin=2 and down_margin=12
                np.arange(48, 88),
                # Estimated col disparity is [-5, 5]
                # So left_margin=7 and right_margin=7
                np.arange(43, 68),
                id="range_row=5, range_col=5 and window_size=5",
            ),
            pytest.param(
                {
                    "row": {"first": 100, "last": 112},
                    "col": {"first": 75, "last": 90},
                },
                7,
                3,
                3,
                [-12, 2],
                [-1, 5],
                # Estimated row disparity is [-12, 2]
                # So up_margin=13 and down_margin=3
                np.arange(87, 116),
                # Estimated col disparity is [-1, 5]
                # So left_margin=2 and right_margin=6
                np.arange(73, 97),
                id="range_row=7, range_col=3 and window_size=3",
            ),
            pytest.param(
                {
                    "row": {"first": 212, "last": 230},
                    "col": {"first": 310, "last": 315},
                },
                5,
                7,
                3,
                [-6, 4],
                [-8, 6],
                # Estimated row disparity is [-6, 4]
                # So up_margin=7 and down_margin=5
                np.arange(205, 236),
                # Estimated col disparity is [-8, 6]
                # So left_margin=9 and right_margin=7
                np.arange(301, 323),
                id="range_row=5, range_col=7 and window_size=3",
            ),
            pytest.param(
                {
                    "row": {"first": 118, "last": 125},
                    "col": {"first": 202, "last": 207},
                },
                9,
                9,
                7,
                [-6, 12],
                [-12, 6],
                # Estimated row disparity is [-6, 12]
                # So up_margin=9 and down_margin=15
                np.arange(109, 141),
                # Estimated col disparity is [-12, 6]
                # So left_margin=15 and right_margin=9
                np.arange(187, 217),
                id="range_row=9, range_col=9 and window_size=7",
            ),
        ],
    )
    def test_run_estimation_with_roi(
        self, estimation_cfg, roi, estimated_d_row, estimated_d_col, expected_row, expected_col
    ):
        """
        Description: Test pipeline with estimation and ROI and check if image coordinates are correct.
        """

        estimation_cfg["ROI"] = roi

        pandora2d_machine = Pandora2DMachine()

        # Check estimation configuration
        checked_cfg = check_conf(estimation_cfg, pandora2d_machine)

        # Get ROI margins
        checked_cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()

        # Create image datasets
        image_datasets = create_datasets_from_inputs(
            input_config=checked_cfg["input"],
            roi=checked_cfg["ROI"],
            estimation_cfg=estimation_cfg["pipeline"].get("estimation"),
        )

        # Run estimation
        pandora2d_machine.run_prepare(image_datasets.left, image_datasets.right, checked_cfg)
        pandora2d_machine.run("estimation", checked_cfg)

        img_shape = pandora2d_machine.left_img["im"].shape

        # Check coordinates
        np.testing.assert_array_equal(pandora2d_machine.left_img.row.values, expected_row)
        np.testing.assert_array_equal(pandora2d_machine.left_img.col.values, expected_col)
        # Check disparities
        np.testing.assert_array_equal(
            pandora2d_machine.left_img.row_disparity.sel(band_disp="min").data, np.full(img_shape, estimated_d_row[0])
        )
        np.testing.assert_array_equal(
            pandora2d_machine.left_img.row_disparity.sel(band_disp="max").data, np.full(img_shape, estimated_d_row[1])
        )
        np.testing.assert_array_equal(
            pandora2d_machine.left_img.col_disparity.sel(band_disp="min").data, np.full(img_shape, estimated_d_col[0])
        )
        np.testing.assert_array_equal(
            pandora2d_machine.left_img.col_disparity.sel(band_disp="max").data, np.full(img_shape, estimated_d_col[1])
        )


class TestDeformationGridMode:
    """
    Test deformation grid mode
    """

    @pytest.fixture()
    def configuration(self, correct_input_cfg, correct_pipeline_without_refinement, init_pixel_conv_grid, tmp_path):
        return {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            **{"output": {"path": str(tmp_path), "deformation_grid": {"init_pixel_conv_grid": init_pixel_conv_grid}}},
        }

    @pytest.mark.parametrize("init_pixel_conv_grid", [[0, 0], [0.5, 0.5]])
    def test_deformation_grid_pipeline(self, configuration, run_pipeline, tmp_path):
        """
        Test execution of a pipeline with deformation grid mode enabled
        """

        run_pipeline(configuration)

        with rasterio.open(tmp_path / "disparity_map" / "row_deformation_map.tif") as src:
            row_deformation_map = src.read(1)
        with rasterio.open(tmp_path / "disparity_map" / "col_deformation_map.tif") as src:
            col_deformation_map = src.read(1)

        # Checking that resulting deformation grids are not full of nans
        assert not np.all(np.isnan(row_deformation_map))
        assert not np.all(np.isnan(col_deformation_map))


class TestDataSamplesOutputConfigReusability:  # pylint: disable=too-few-public-methods
    """
    Test that output configurations from data_samples pipelines can be re-executed.
    """

    @pytest.mark.parametrize("config_data", filelist_parametrize_generator())
    def test_output_config_can_be_reused(self, run_pipeline, tmp_path, config_data, cones_support_files):
        """
        Description: Check that each output configuration generated from data_samples can be run again.

        Runs each data_samples JSON config with cones/monoband images (support files generated
        in tmp_path). Verifies that the output config.json can be re-executed as a second run.

        Note: Estimation pipelines (an_estimation_pipeline.json) are expected to fail on re-execution
        because the keys ``estimated_shifts``, ``phase_diff`` and ``error`` written into the output
        configuration are not valid inputs for the estimation schema.
        """
        configuration, config_file = config_data

        # We skip confidence pipeline on Windows due to known access violation in compute_ambiguity pandora method.
        # This will be removed after completing issue 460.
        if sys.platform.startswith("win") and "confidence" in config_file.name:
            pytest.skip("Skipping confidence pipeline on Windows")

        configuration = transform_config_to_cones(configuration, cones_support_files)

        output_dir = tmp_path / config_file.stem
        configuration["output"]["path"] = str(output_dir)

        run_pipeline(configuration)

        output_config_path = output_dir / "config.json"
        assert output_config_path.exists()

        if is_estimation_pipeline(config_file):
            # Known limitation: estimation output configs cannot be re-run because the keys
            # estimated_shifts, phase_diff and error written into the output config are not
            # valid inputs for the estimation schema.
            with pytest.raises(MissKeyCheckerError, match=r"Missing keys in expected schema.*estimated_shifts"):
                pandora2d.main(output_config_path, verbose=False)
        else:
            pandora2d.main(output_config_path, verbose=False)
