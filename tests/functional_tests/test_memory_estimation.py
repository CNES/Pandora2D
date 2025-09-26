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
