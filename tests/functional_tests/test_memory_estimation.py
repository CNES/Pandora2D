#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Memory estimation tests."""

import json
from typing import List
from copy import deepcopy

import numpy as np
import rasterio
import pytest

from pandora2d import Pandora2DMachine, memory_estimation
from pandora2d.check_configuration import check_conf

# pylint: disable=redefined-outer-name,too-few-public-methods,invalid-name,too-many-arguments,
# pylint: disable=too-many-positional-arguments


class TestEstimateTotalMemoryConsumption:
    """Test the estimation of memory consumption."""

    @pytest.fixture(scope="class")
    def result_store(self, request, tmp_path_factory):
        """Yield a list of objects that will be dumped to JSON file."""
        store: List = []

        yield store

        tmp_directory = tmp_path_factory.mktemp("test_reports")
        # Replace ":" by "#" because "::" is not accepted for a Windows path.
        directory = tmp_directory / request.node.nodeid.replace(":", "#")
        directory.mkdir(parents=True)
        file_path = directory / "memory_report.json"
        with file_path.open("w") as memory_report:
            json.dump(store, memory_report)

    @pytest.fixture
    def state_machine(self):
        """Instantiate a Pandora2D state machine."""
        return Pandora2DMachine()

    @pytest.fixture
    def checked_config(self, config, state_machine):
        """Run check_conf on config and return the result."""
        return check_conf(config, state_machine)

    @pytest.fixture
    def config(
        self,
        tmp_path,
        left_img_path,
        right_img_path,
        row_disparity,
        col_disparity,
        matching_cost_method,
        window_size,
        step,
        subpix,
    ):
        """Config."""
        return {
            "input": {
                "left": {
                    "img": left_img_path,
                    "nodata": -9999,
                },
                "right": {
                    "img": right_img_path,
                    "nodata": -9999,
                },
                "row_disparity": row_disparity,
                "col_disparity": col_disparity,
            },
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": matching_cost_method,
                    "window_size": window_size,
                    "step": step,
                    "subpix": subpix,
                },
                "disparity": {
                    "disparity_method": "wta",
                    "invalid_disparity": -9999,
                },
            },
            "output": {
                "path": str(tmp_path),
            },
        }

    @pytest.fixture
    def add_roi_to_config(self, config, roi):
        """Add roi to config."""
        config["ROI"] = roi
        return config

    # Warning: we must stay in a case where we are not above max memory to measure the same thing that we estimate
    @pytest.fixture
    def measured_consumption(self, config, run_pipeline, MemoryTracer):
        """Run pandora2d with config and measure memory consumption."""
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            run_pipeline(config)
        return memory_tracer

    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "sad"])
    @pytest.mark.parametrize("step", [[1, 1], [1, 4], [4, 1]])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test(
        self, checked_config, state_machine, measured_consumption, result_store, matching_cost_method, step, subpix
    ):
        """Test estimate_total_consumption without ROI."""
        height, width = memory_estimation.compute_effective_image_size(
            checked_config, state_machine.margins_img.global_margins
        )
        estimation = memory_estimation.estimate_total_consumption(
            checked_config, height, width, state_machine.margins_disp.global_margins
        )

        result_store.append(
            {
                "method": matching_cost_method,
                "step": step,
                "subpix": subpix,
                "estimation": float(estimation),  # cast in order to be json serializable
                "current": measured_consumption.current,
                "peak": measured_consumption.peak,
            }
        )
        assert estimation == pytest.approx(
            measured_consumption.current,
            rel=memory_estimation.RELATIVE_ESTIMATION_MARGIN,
        )

    @pytest.mark.usefixtures("add_roi_to_config")
    @pytest.mark.parametrize("matching_cost_method", ["zncc_python", "mutual_information", "sad", "zncc"])
    @pytest.mark.parametrize("step", [[1, 1], [1, 4], [4, 1]])
    @pytest.mark.parametrize("subpix", [1, 4])
    @pytest.mark.parametrize(
        "roi",
        [
            pytest.param(
                {"row": {"first": 0, "last": 150}, "col": {"first": 60, "last": 150}},
                id="Small",
            ),
            pytest.param(
                {"row": {"first": 0, "last": 200}, "col": {"first": 0, "last": 310}},
                id="Larger",
            ),
        ],
    )
    def test_with_roi(
        self, checked_config, state_machine, measured_consumption, result_store, matching_cost_method, step, subpix
    ):
        """Test estimate_total_consumption with ROI."""

        height, width = memory_estimation.compute_effective_image_size(
            checked_config, state_machine.margins_img.global_margins
        )
        estimation = memory_estimation.estimate_total_consumption(
            checked_config, height, width, state_machine.margins_disp.global_margins
        )

        result_store.append(
            {
                "method": matching_cost_method,
                "step": step,
                "subpix": subpix,
                "estimation": float(estimation),  # cast in order to be json serializable
                "current": measured_consumption.current,
                "peak": measured_consumption.peak,
            }
        )
        assert estimation == pytest.approx(
            measured_consumption.current,
            rel=memory_estimation.RELATIVE_ESTIMATION_MARGIN,
        )


class TestSegmentMode:
    """
    Test execution of pandora2d pipelines with segment mode
    """

    @pytest.fixture()
    def configuration(self, input_cfg, pipeline, request, tmp_path):
        """
        Pandora2d configuration
        """

        return {
            **request.getfixturevalue(input_cfg),
            **request.getfixturevalue(pipeline),
            **{"output": {"path": str(tmp_path)}},
        }

    @pytest.fixture()
    def memory_per_work(self):
        """
        Define memory per work in MB
        """
        return 20

    @pytest.fixture()
    def configuration_segment(self, configuration, memory_per_work):
        """
        Pandora2d configuration with segment mode
        """

        configuration_segment = deepcopy(configuration)
        configuration_segment["segment_mode"] = {"enable": True, "memory_per_work": memory_per_work}
        return configuration_segment

    @pytest.fixture
    def open_results(self):
        """
        Get result disparity maps and configuration
        """

        def inner(path):

            with rasterio.open(path / "disparity_map" / "row_map.tif") as src:
                row_map = src.read(1)
            with rasterio.open(path / "disparity_map" / "col_map.tif") as src:
                col_map = src.read(1)
            with rasterio.open(path / "disparity_map" / "correlation_score.tif") as src:
                correlation_score = src.read(1)
            with rasterio.open(path / "disparity_map" / "validity.tif") as src:
                validity = src.read()
            with open(path / "disparity_map" / "attributes.json", encoding="utf8") as attributes_file:
                output_attributes = json.load(attributes_file)
            with open(path / "disparity_map" / "report.json", encoding="utf8") as report_file:
                output_report = json.load(report_file)
            with open(path / "config.json", encoding="utf8") as cfg_file:
                output_config = json.load(cfg_file)

            return (
                row_map,
                col_map,
                correlation_score,
                validity,
                output_attributes,
                output_report,
                output_config,
            )

        return inner

    @pytest.mark.parametrize(
        ["input_cfg", "pipeline", "matching_cost_method", "subpix", "memory_per_work"],
        [
            pytest.param(
                "correct_input_cfg",
                "correct_pipeline_without_refinement",
                "mutual_information",
                1,
                20,
                id="Pipeline without refinement, mutual information and no enough memory without segment mode",
            ),
            pytest.param(
                "correct_input_with_left_mask",
                "correct_pipeline_with_dichotomy_cpp",
                "zncc_python",
                1,
                20,
                id="Pipeline with dichotomy cpp, zncc python, mask and no enough memory without segment mode",
            ),
            pytest.param(
                "correct_input_with_right_mask",
                "correct_pipeline_without_refinement",
                "zncc_python",
                1,
                400,
                id="Pipeline without refinement, zncc_python, mask and enough memory",
            ),
            pytest.param(
                "correct_input_cfg",
                "correct_pipeline_without_refinement",
                "zncc",
                2,
                40,
                id="Pipeline without refinement, zncc cpp, subpix=2 and no enough memory without segment mode",
            ),
        ],
    )
    def test_segment_mode(
        self,
        open_results,
        configuration,
        configuration_segment,
        MemoryTracer,
        run_pipeline,
        matching_cost_method,
        subpix,
    ):  # pylint: disable=unused-argument
        """
        Description: Check that pandora2d executions with and without segment modes give the same result
        """

        # Run basic configuration
        path = run_pipeline(configuration)

        row_map, col_map, correlation_score, validity, output_attributes, output_report, output_cfg = open_results(path)

        # Run segment mode configuration
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            path_segment = run_pipeline(configuration_segment)

        (
            row_map_segment,
            col_map_segment,
            correlation_score_segment,
            validity_segment,
            output_attributes_segment,
            output_report_segment,
            output_cfg_segment,
        ) = open_results(path_segment)

        np.testing.assert_array_equal(row_map, row_map_segment)
        np.testing.assert_array_equal(col_map, col_map_segment)
        np.testing.assert_array_equal(correlation_score, correlation_score_segment)
        np.testing.assert_array_equal(validity, validity_segment)

        # Delete segment_mode part of the configuration because this is the only part that must be
        # different between the two dictionaries.
        del output_cfg["segment_mode"]
        del output_cfg_segment["segment_mode"]
        assert output_attributes == output_attributes_segment
        assert output_report == output_report_segment
        assert output_cfg == output_cfg_segment
        assert memory_tracer.current <= configuration_segment["segment_mode"]["memory_per_work"]

    @pytest.mark.parametrize(
        ["input_cfg", "pipeline", "roi", "matching_cost_method", "subpix", "memory_per_work"],
        [
            pytest.param(
                "correct_input_cfg",
                "correct_pipeline_without_refinement",
                {"col": {"first": 100, "last": 350}, "row": {"first": 200, "last": 450}},
                "zncc_python",
                1,
                20,
                id="251x251 ROI, 20 MB memory per work, zncc python, without refinement",
            ),
            pytest.param(
                "correct_input_with_left_right_mask",
                "correct_pipeline_with_dichotomy_cpp",
                {"col": {"first": 0, "last": 50}, "row": {"first": 0, "last": 50}},
                "mutual_information",
                1,
                2,
                id="51x51 ROI, 2 MB memory per work, mask, mutual_information with refinement",
            ),
            pytest.param(
                "correct_input_cfg",
                "correct_pipeline_without_refinement",
                {"col": {"first": 200, "last": 350}, "row": {"first": 200, "last": 350}},
                "zncc_python",
                1,
                10,
                id="151x151 ROI, 10 MB memory per work, zncc python without refinement",
            ),
            pytest.param(
                "correct_input_with_left_mask",
                "correct_pipeline_with_dichotomy_cpp",
                {"col": {"first": 200, "last": 300}, "row": {"first": 100, "last": 200}},
                "zncc",
                1,
                15,
                id="101x101 ROI, 15 MB memory per work, mask, zncc cpp with refinement",
            ),
            pytest.param(
                "correct_input_cfg",
                "correct_pipeline_without_refinement",
                {"col": {"first": 200, "last": 350}, "row": {"first": 200, "last": 350}},
                "mutual_information",
                2,
                10,
                id="151x151 ROI, 10 MB memory per work, subpix=2, mutual information without refinement",
            ),
            pytest.param(
                "correct_input_with_right_mask",
                "correct_pipeline_without_refinement",
                {"col": {"first": 0, "last": 50}, "row": {"first": 0, "last": 50}},
                "zncc_python",
                4,
                15,
                id="51x51 ROI, 15 MB memory per work, mask, subpix=4, zncc python without refinement",
            ),
            pytest.param(
                "correct_input_with_right_mask",
                "correct_pipeline_without_refinement",
                {"col": {"first": 0, "last": 50}, "row": {"first": 0, "last": 50}},
                "zncc",
                4,
                15,
                id="51x51 ROI, 15 MB memory per work, mask, subpix=4, zncc cpp without refinement",
            ),
            pytest.param(
                "correct_input_with_left_right_mask",
                "correct_pipeline_with_dichotomy_cpp",
                {"col": {"first": 0, "last": 50}, "row": {"first": 0, "last": 50}},
                "mutual_information",
                1,
                200,
                id="51x51 ROI, mask, mutual_information and enough memory",
            ),
        ],
    )
    def test_segment_mode_with_roi(
        self,
        open_results,
        roi,
        configuration,
        configuration_segment,
        MemoryTracer,
        run_pipeline,
        matching_cost_method,
        subpix,
    ):  # pylint: disable=unused-argument
        """
        Description: Check that pandora2d executions with and without segment modes and with a ROI give the same result
        """

        configuration["ROI"] = roi
        configuration_segment["ROI"] = roi

        # Run basic configuration
        path = run_pipeline(configuration)

        row_map, col_map, correlation_score, validity, output_attributes, output_report, output_cfg = open_results(path)

        # Run segment mode configuration
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            path_segment = run_pipeline(configuration_segment)

        (
            row_map_segment,
            col_map_segment,
            correlation_score_segment,
            validity_segment,
            output_attributes_segment,
            output_report_segment,
            output_cfg_segment,
        ) = open_results(path_segment)

        np.testing.assert_array_equal(row_map, row_map_segment)
        np.testing.assert_array_equal(col_map, col_map_segment)
        np.testing.assert_array_equal(correlation_score, correlation_score_segment)
        np.testing.assert_array_equal(validity, validity_segment)

        # Delete segment_mode part of the configuration because this is the only part that must be
        # different between the two dictionaries.
        del output_cfg["segment_mode"]
        del output_cfg_segment["segment_mode"]
        assert output_attributes == output_attributes_segment
        assert output_report == output_report_segment
        assert output_cfg == output_cfg_segment
        assert memory_tracer.current <= configuration_segment["segment_mode"]["memory_per_work"]
