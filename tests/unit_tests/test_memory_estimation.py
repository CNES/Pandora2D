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
This file contains unit tests associated to the pandora2d memory estimation
"""

# pylint: disable=redefined-outer-name,too-few-public-methods,invalid-name,too-many-lines

from typing import Dict, Tuple, cast
from copy import deepcopy

import numpy as np
import pandora
import pytest

from pandora2d import common, criteria, memory_estimation
from pandora2d.check_configuration import check_conf, check_roi_section
from pandora2d.criteria import get_criteria_dataarray
from pandora2d.img_tools import (
    create_datasets_from_inputs,
    get_roi_processing,
    shift_subpix_img_2d,
)
from pandora2d.margins import Margins, NullMargins
from pandora2d.matching_cost import CorrelationMethods
from pandora2d.state_machine import Pandora2DMachine


@pytest.fixture()
def input_config(correct_input_cfg, random_left_image_path, random_right_image_path):
    """Input section of the configuration file with different disparity ranges for rows and columns."""
    correct_input_cfg["input"]["left"]["img"] = str(random_left_image_path)
    correct_input_cfg["input"]["right"]["img"] = str(random_right_image_path)
    correct_input_cfg["input"]["row_disparity"] = {"init": 1, "range": 3}
    correct_input_cfg["input"]["col_disparity"] = {"init": 1, "range": 2}
    return correct_input_cfg


@pytest.fixture()
def image_datasets(input_config):
    """Left and right images according to input section of the configuration file."""
    return create_datasets_from_inputs(input_config["input"])


class TestInputSize:
    """
    Test methods linked to input size computation
    """

    @pytest.mark.parametrize(
        ["roi", "expected"],
        [
            pytest.param(
                None,
                (375, 450),
                id="No ROI",
            ),
            pytest.param(
                {"row": {"first": 0, "last": 10}, "col": {"first": 0, "last": 20}},
                (11, 21),
                id="ROI of size (11,21)",
            ),
            pytest.param(
                {"row": {"first": 371, "last": 380}, "col": {"first": 0, "last": 2}},
                (4, 3),
                id="ROI of size (4,3)",
            ),
        ],
    )
    def test_get_img_size(self, left_img_path, roi, expected):
        """
        Test the get_img_size method
        """

        assert memory_estimation.get_img_size(left_img_path, roi=roi) == expected

    def test_raises_error_with_roi_outside_image(self, left_img_path):
        """
        Test that the get_img_size method raises an error with a ROI outside the image
        """

        roi = {"row": {"first": 379, "last": 380}, "col": {"first": 470, "last": 480}}

        with pytest.raises(ValueError, match="Roi specified is outside the image"):
            memory_estimation.get_img_size(left_img_path, roi)

    @pytest.mark.parametrize(
        ["disparity", "before_margins", "after_margins", "subpix", "expected"],
        [
            pytest.param(
                {"init": 0, "range": 2},
                0,
                0,
                1,
                5,
                id="Centered disparities",
            ),
            pytest.param(
                {"init": 3, "range": 1},
                0,
                0,
                1,
                3,
                id="Positive disparity",
            ),
            pytest.param(
                {"init": -5, "range": 4},
                0,
                0,
                1,
                9,
                id="Negative disparity",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                2,
                0,
                1,
                7,
                id="Centered disparities with before margins",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                0,
                4,
                1,
                9,
                id="Centered disparities with after margins",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                0,
                0,
                4,
                17,
                id="Centered disparities with subpix",
            ),
        ],
    )
    def test_get_nb_disp(self, disparity, before_margins, after_margins, subpix, expected):
        """
        Test the get_nb_disp method
        """

        assert memory_estimation.get_nb_disp(disparity, before_margins, after_margins, subpix) == expected

    @pytest.mark.parametrize(
        ["disparity", "expected"],
        [
            pytest.param(
                "correct_grid",
                14,
            ),
            pytest.param(
                "second_correct_grid",
                37,
            ),
        ],
    )
    def test_get_nb_disp_with_grid(self, disparity, expected, request):
        """
        Test the get_nb_disp method with grid disparities
        """

        assert memory_estimation.get_nb_disp(request.getfixturevalue(disparity)) == expected

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity", "global_margins", "expected"],
        [
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 3},
                Margins(0, 0, 0, 0),
                Margins(3, 2, 3, 2),
                id="Centered disparities and initial margins=(0,0,0,0)",
            ),
            pytest.param(
                {"init": 0, "range": 1},
                {"init": 0, "range": 1},
                Margins(3, 3, 3, 3),
                Margins(4, 4, 4, 4),
                id="Initial margins greater than disparities",
            ),
            pytest.param(
                {"init": 3, "range": 2},
                {"init": 0, "range": 1},
                Margins(1, 2, 3, 4),
                Margins(2, 1, 4, 9),
                id="Positive row disparity",
            ),
            pytest.param(
                {"init": 0, "range": 1},
                {"init": -4, "range": 2},
                Margins(1, 2, 3, 4),
                Margins(7, 3, 1, 5),
                id="Negative col disparity",
            ),
        ],
    )
    def test_get_roi_margins(self, row_disparity, col_disparity, global_margins, expected):
        """
        Test the get_roi_margins method
        """

        assert memory_estimation.get_roi_margins(row_disparity, col_disparity, global_margins) == expected

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity", "global_margins", "expected"],
        [
            pytest.param(
                "correct_grid",
                "second_correct_grid",
                Margins(0, 0, 0, 0),
                Margins(26, 5, 10, 8),
                id="Grid disparities and initial margins=(0,0,0,0)",
            ),
            pytest.param(
                "correct_grid",
                "second_correct_grid",
                Margins(1, 2, 3, 4),
                Margins(27, 7, 13, 12),
                id="Grid disparities and initial margins=(1,2,3,4)",
            ),
        ],
    )
    def test_get_roi_margins_with_grid(self, row_disparity, col_disparity, global_margins, expected, request):
        """
        Test the get_roi_margins method with grid disparities
        """

        assert (
            memory_estimation.get_roi_margins(
                request.getfixturevalue(row_disparity), request.getfixturevalue(col_disparity), global_margins
            )
            == expected
        )

    @pytest.mark.parametrize(
        ["height", "width", "sum_nb_bytes", "expected"],
        [
            pytest.param(
                375,
                450,
                22,
                3.540,
            ),
            pytest.param(
                10,
                20,
                20,
                0.004,
            ),
        ],
    )
    def test_img_dataset_size(self, height, width, sum_nb_bytes, expected):
        """
        Test the img_dataset_size method
        """

        assert pytest.approx(memory_estimation.img_dataset_size(height, width, sum_nb_bytes), abs=1e-3) == expected

    @pytest.mark.parametrize(
        ["height", "width", "data_vars", "expected"],
        [
            pytest.param(
                375,
                450,
                ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max", "msk"],
                3.54,
                id="6 data variables and image 375x450",
            ),
            pytest.param(
                10,
                20,
                ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max", "msk"],
                0.0038,
                id="5 data variables and image 10x20",
            ),
            pytest.param(
                1000,
                1000,
                ["im", "im", "im", "im"],
                15.259,
                id="4 data variables and image 1000x1000",
            ),
            pytest.param(
                10000,
                5000,
                ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max", "msk"],
                1049.042,
                id="6 data variables and image 10000x5000",
            ),
        ],
    )
    def test_input_size(self, height, width, data_vars, expected):
        """
        Test the input_size method
        """

        assert pytest.approx(memory_estimation.estimate_input_size(height, width, data_vars), abs=1e-3) == expected

    def test_memory_input(self, MemoryTracer, correct_input_cfg):
        """
        Test that the value returned by input_size corresponds to the memory occupied by the image datasets.
        """

        # Memory computed by input_size method
        height, width = memory_estimation.get_img_size(correct_input_cfg["input"]["left"]["img"])
        data_variables = memory_estimation.IMG_DATA_VAR

        # We have a factor of x2 for left and right images
        memory_computed = 2 * memory_estimation.estimate_input_size(height, width, data_variables)

        # Memory consumed when creating the two images datasets
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            image_datasets = create_datasets_from_inputs(correct_input_cfg["input"])

        # Check that the estimated image dataset memory corresponds to the measured memory within 10%.
        assert memory_computed == pytest.approx(memory_tracer.current, rel=0.10)
        # Check that the estimated image dataset memory corresponds to the result of
        # image_dataset.left.nbytes +  image_dataset.right.nbytes
        image_dataset_nbytes = (image_datasets.left.nbytes + image_datasets.right.nbytes) / memory_estimation.BYTE_TO_MB
        assert memory_computed == pytest.approx(image_dataset_nbytes, rel=0.05)

    def test_memory_input_with_roi(self, MemoryTracer, correct_input_cfg, correct_roi_sensor, correct_pipeline):
        """
        Test that the value returned by input_size corresponds to the memory occupied by the image datasets
        with a ROI.
        """

        # Add ROI to user cfg
        user_cfg = {**correct_input_cfg, **correct_roi_sensor, **correct_pipeline, "output": {"path": "memory_output"}}

        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(user_cfg, pandora2d_machine)
        # Get roi processing to call create_dataset_from_inputs with ROI
        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        # Memory computed by input_size method
        height, width = memory_estimation.get_img_size(correct_input_cfg["input"]["left"]["img"], roi=cfg["ROI"])
        roi_margins = memory_estimation.get_roi_margins(
            correct_input_cfg["input"]["row_disparity"],
            correct_input_cfg["input"]["col_disparity"],
            pandora2d_machine.margins_img.global_margins,
        )
        # Final height and width are ROI size + margins
        height += roi_margins.up + roi_margins.down
        width += roi_margins.left + roi_margins.right

        data_variables = memory_estimation.IMG_DATA_VAR

        # We have a factor of x2 for left and right images
        memory_computed = 2 * memory_estimation.estimate_input_size(height, width, data_variables)

        # Memory consumed when creating the two images datasets
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            image_datasets = create_datasets_from_inputs(cfg["input"], roi)

        # Check that the estimated image dataset memory corresponds to the measured memory within 25%.
        # Estimated dataset size is 0.37 and measured dataset size is 0.39.
        # Total measured memory is 0.47.
        assert memory_computed == pytest.approx(memory_tracer.current, rel=0.25)
        # Check that the estimated image dataset memory corresponds to the result of
        # image_dataset.left.nbytes +  image_dataset.right.nbytes
        image_dataset_nbytes = (image_datasets.left.nbytes + image_datasets.right.nbytes) / memory_estimation.BYTE_TO_MB
        assert memory_computed == pytest.approx(image_dataset_nbytes, rel=0.05)


class TestCostVolumesSize:
    """
    Test methods linked to cost volumes size computation
    """

    @pytest.mark.parametrize("size", [(375, 450)])  # (height, width)
    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity", "step", "subpix", "margins_disp", "data_vars", "expected"],
        [
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 2},
                [1, 1],
                1,
                Margins(0, 0, 0, 0),
                ["cost_volumes_float", "criteria"],
                20.117,
                id="Float cost volumes",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 2},
                [1, 1],
                1,
                Margins(0, 0, 0, 0),
                ["cost_volumes_double", "criteria"],
                36.210,
                id="Double cost volumes",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 2},
                [2, 3],
                1,
                Margins(0, 0, 0, 0),
                ["cost_volumes_float", "criteria"],
                3.362,
                id="Step=[2,3]",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 2},
                [1, 1],
                4,
                Margins(0, 0, 0, 0),
                ["cost_volumes_float", "criteria"],
                232.547,
                id="Subpix=4",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 2},
                [1, 1],
                1,
                Margins(1, 2, 3, 4),
                ["cost_volumes_float", "criteria"],
                79.662,
                id="Margins_disp=(1,2,3,4)",
            ),
            pytest.param(
                {"init": 2, "range": 1},
                {"init": -6, "range": 3},
                [1, 1],
                1,
                Margins(0, 0, 0, 0),
                ["cost_volumes_float", "criteria"],
                16.898,
                id="Disp row different from disp col",
            ),
            pytest.param(
                {"init": 2, "range": 1},
                {"init": -6, "range": 3},
                [2, 1],
                4,
                Margins(1, 3, 2, 5),
                ["cost_volumes_float", "criteria"],
                611.964,
                id="Combinaison of parameters",
            ),
        ],
    )
    def test_cost_volumes_size(
        self, matching_cost_config, size, col_disparity, row_disparity, margins_disp, data_vars, expected
    ):
        """
        Test the cost_volumes_size method
        """

        user_cfg = {
            "input": {"col_disparity": col_disparity, "row_disparity": row_disparity},
            "pipeline": {"matching_cost": matching_cost_config},
        }

        assert (
            pytest.approx(
                memory_estimation.estimate_cost_volumes_size(user_cfg, size[0], size[1], margins_disp, data_vars),
                abs=1e-3,
            )
            == expected
        )

    def get_cv_coords(self, left_image, cfg, matching_cost_instance, pandora2d_machine):
        """
        Get cost volumes coordinates to call allocate_cost_volumes method
        """

        img_row_coordinates = left_image["im"].coords["row"].values
        img_col_coordinates = left_image["im"].coords["col"].values

        row_coords, col_coords = matching_cost_instance.get_cv_row_col_coords(
            img_row_coordinates, img_col_coordinates, cfg
        )
        # Get disparity coordinates for cost_volumes
        disps_row_coords = matching_cost_instance.get_disp_row_coords(
            left_image, pandora2d_machine.margins_disp.global_margins
        )
        disps_col_coords = matching_cost_instance.get_disp_col_coords(
            left_image, pandora2d_machine.margins_disp.global_margins
        )

        return row_coords, col_coords, disps_row_coords, disps_col_coords

    def get_cv_attributes(self, left_image, matching_cost_cfg, pandora2d_machine):
        """
        Get cost volumes attributes to call allocate_cost_volumes method
        """

        grid_attrs = left_image.attrs

        grid_attrs.update(
            {
                "window_size": matching_cost_cfg["window_size"],
                "subpixel": matching_cost_cfg["subpix"],
                "offset_row_col": int((matching_cost_cfg["window_size"] - 1) / 2),
                "measure": matching_cost_cfg["matching_cost_method"],
                "type_measure": "max",
                "disparity_margins": pandora2d_machine.margins_disp.global_margins,
                "step": matching_cost_cfg["step"],
            }
        )

        return grid_attrs

    @pytest.fixture()
    def user_cfg_cv_memory(self, correct_input_cfg, matching_cost_config):
        """
        User configuration to test cost volumes memory estimation
        """

        user_cfg = {
            **correct_input_cfg,
            "pipeline": {
                "matching_cost": matching_cost_config,
                "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            },
            "output": {"path": "memory_cv_output"},
        }

        return user_cfg

    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [3, 3]])
    def test_memory_cost_volumes(self, MemoryTracer, user_cfg_cv_memory, matching_cost_object):
        """
        Test that the value returned by cost_volumes_size corresponds to the memory occupied by the cost volume dataset.
        """

        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(user_cfg_cv_memory, pandora2d_machine)

        # Compute cost volumes size estimation
        height, width = memory_estimation.get_img_size(cfg["input"]["left"]["img"])
        memory_computed = memory_estimation.estimate_cost_volumes_size(
            cfg, height, width, pandora2d_machine.margins_disp.global_margins, memory_estimation.CV_FLOAT_DATA_VAR
        )

        image_datasets = create_datasets_from_inputs(cfg["input"])

        matching_cost_ = matching_cost_object(cfg["pipeline"]["matching_cost"])

        # Get cost volumes coordinates and attributes
        row_coords, col_coords, disps_row_coords, disps_col_coords = self.get_cv_coords(
            image_datasets.left, cfg, matching_cost_, pandora2d_machine
        )
        grid_attrs = self.get_cv_attributes(image_datasets.left, cfg["pipeline"]["matching_cost"], pandora2d_machine)

        # Memory consumed when allocating the 4D cost volumes dataset
        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            cost_volumes = matching_cost_.allocate_cost_volumes(
                grid_attrs, row_coords, col_coords, disps_row_coords, disps_col_coords, None
            )
            cost_volumes["criteria"] = get_criteria_dataarray(image_datasets.left, image_datasets.right, cost_volumes)

        # Check that the estimated image dataset memory corresponds to the measured memory within 5%.
        assert memory_computed == pytest.approx(memory_tracer.current, rel=0.05)
        # Check that the estimated cost volumes dataset memory corresponds to the result of cost_volumes.nbytes
        cv_nbytes = (cost_volumes.nbytes) / memory_estimation.BYTE_TO_MB
        assert memory_computed == pytest.approx(cv_nbytes, rel=0.05)


class TestPandoraCostVolumesSize:
    """
    Test methods linked to pandora's cost volumes size computation
    """

    @pytest.fixture()
    def matching_cost_config(self, subpix, step):
        """Matching cost section of the configuration file."""
        return {
            "matching_cost_method": "ssd",
            "window_size": 3,
            "subpix": subpix,
            "step": step,
        }

    @pytest.fixture()
    def config(self, input_config, matching_cost_config):
        """Full configuration."""
        return {
            **input_config,
            "pipeline": {"matching_cost": matching_cost_config},
        }

    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 4]])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize("margins", [NullMargins(), Margins(1, 2, 3, 4)])
    def test(self, MemoryTracer, image_datasets, config, margins):
        """Test that cost volumes size computation works as expected."""

        height, width = image_datasets.left.sizes["row"], image_datasets.left.sizes["col"]

        pandora_matching_cost_config = deepcopy(config["pipeline"]["matching_cost"])
        pandora_matching_cost_config["step"] = config["pipeline"]["matching_cost"]["step"][1]

        pandora_matching_cost = pandora.matching_cost.AbstractMatchingCost(  # type: ignore[abstract]
            **pandora_matching_cost_config
        )

        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            cost_volume = pandora_matching_cost.allocate_cost_volume(
                image_datasets.left,
                (
                    image_datasets.left["col_disparity"].sel(band_disp="min").data - margins.left,
                    image_datasets.left["col_disparity"].sel(band_disp="max").data + margins.right,
                ),
                config,
            )
        estimation = memory_estimation.estimate_pandora_cost_volume_size(config, height, width, margins)

        assert estimation == pytest.approx(
            cost_volume["cost_volume"].data.nbytes / memory_estimation.BYTE_TO_MB, rel=0.05
        )
        assert estimation == pytest.approx(memory_tracer.current, rel=0.05, abs=1e-2)


class TestShiftedRightImages:
    """Test memory consumption of shifted right images."""

    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test(self, MemoryTracer, image_datasets, subpix):
        """Test memory consumption of shifted right images."""

        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            images = shift_subpix_img_2d(image_datasets.right, subpix)
        # We exclude the first image from the count as it is excluded in estimate_shifted_image_datasets.rights_size
        images_size = sum(image.nbytes for image in images[1:]) / memory_estimation.BYTE_TO_MB

        result = memory_estimation.estimate_shifted_right_images_size(
            image_datasets.right.dims["row"], image_datasets.right.dims["col"], subpix
        )

        assert result == pytest.approx(images_size, rel=0.05)
        # When subpix = 1, we approximate with absolute tolerance since we expect a value close to 0,
        # making relative tolerance irrelevant in this case.
        assert result == pytest.approx(memory_tracer.current, rel=0.05, abs=1e-2)


class TestDatasetDispMap:
    """Test memory estimation of dataset disp map."""

    @pytest.fixture()
    def matching_cost_config(self, step):
        """Matching cost section of the configuration file."""
        return {
            "matching_cost_method": "mutual_information",
            "window_size": 3,
            "step": step,
            "subpix": 1,
        }

    @pytest.fixture()
    def config(self, input_config, matching_cost_config):
        """Full configuration."""
        return {
            **input_config,
            "pipeline": {"matching_cost": matching_cost_config},
        }

    @pytest.mark.parametrize("dtype_argument", [np.float32, "float32"])
    @pytest.mark.parametrize("step", [[1, 1], [1, 2], [2, 1]])
    @pytest.mark.parametrize("image_size", [(200, 300), (700, 500)])
    def test(self, MemoryTracer, config, step, image_datasets, image_size: Tuple[int, int], dtype_argument):
        """Test coherence between estimated memory consumption and actual memory consumption."""

        matching_cost = CorrelationMethods(config["pipeline"]["matching_cost"])
        matching_cost.allocate(image_datasets.left, image_datasets.right, config["pipeline"]["matching_cost"])
        cost_volumes = matching_cost.cost_volumes

        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            dataset_validity = criteria.get_validity_dataset(cost_volumes["criteria"])
            dataset_disp_maps = common.dataset_disp_maps(
                cost_volumes.cost_volumes.coords,
                dataset_validity,
                {
                    "offset": {
                        "row": config.get("ROI", {}).get("row", {}).get("first", 0),
                        "col": config.get("ROI", {}).get("col", {}).get("first", 0),
                    },
                    "step": {
                        "row": config["pipeline"]["matching_cost"]["step"][0],
                        "col": config["pipeline"]["matching_cost"]["step"][1],
                    },
                    "invalid_disp": -9999,
                    "crs": image_datasets.left.crs,
                    "transform": image_datasets.left.transform,
                },
            )

        estimation = memory_estimation.estimate_dataset_disp_map_size(*image_size, step, dtype_argument)

        assert estimation == pytest.approx(dataset_disp_maps.nbytes / memory_estimation.BYTE_TO_MB, rel=0.05)
        assert estimation == pytest.approx(memory_tracer.current, rel=0.05, abs=1e-2)


class TestSegmentImageByRows:
    """Test segment_image_by_rows."""

    @pytest.fixture
    def state_machine(self):
        """Instantiate a Pandora2D state machine."""
        return Pandora2DMachine()

    @pytest.fixture
    def checked_config(self, config, state_machine):
        """Run check_conf on config and return the result."""
        return check_conf(config, state_machine)

    @pytest.fixture
    def segment_mode(self, memory_per_work):
        return {
            "enable": True,
            "memory_per_work": memory_per_work,
        }

    @pytest.fixture
    def config(self, tmp_path, input_config, segment_mode):
        return {
            **input_config,
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "mutual_information",
                    "window_size": 5,
                    "step": [1, 1],
                    "subpix": 1,
                },
                "disparity": {
                    "disparity_method": "wta",
                    "invalid_disparity": -9999,
                },
            },
            "segment_mode": segment_mode,
            "output": {
                "path": str(tmp_path),
            },
        }

    @pytest.fixture
    def dataset_disp_map_size(self, image_size: Tuple[int, int], checked_config):
        return memory_estimation.estimate_dataset_disp_map_size(
            *image_size,
            checked_config["pipeline"]["matching_cost"]["step"],
            checked_config["pipeline"]["matching_cost"]["float_precision"],
        )

    @pytest.fixture
    def image_can_be_fully_reconstructed(self, image_size: Tuple[int, int]):
        """Helper that checks that the image can be fully reconstructed from an ROI list."""

        def inner(rois):
            total_number_of_rows, total_number_of_columns = image_size
            row_sorted = sorted(rois, key=lambda roi: roi["row"]["first"])
            for first_roi, second_roi in zip(row_sorted[:-1], row_sorted[1:]):
                assert second_roi["row"]["first"] == first_roi["row"]["last"] + 1, "ROIs should be continuous"
                assert first_roi["col"] == second_roi["col"], "ROIs should be only on rows."
            # Following asserts are relevant because we previously checked that all ROIs were contiguous:
            assert row_sorted[0]["row"]["first"] == 0, "ROI should start on the first row"
            assert row_sorted[-1]["row"]["last"] == total_number_of_rows - 1, "ROI should stop on the last row"
            # We previously checked that all columns ROIs were the same thus we can use the first one:
            assert row_sorted[0]["col"]["first"] == 0, "ROI should start on the first col"
            assert row_sorted[0]["col"]["last"] == total_number_of_columns - 1, "ROI should stop on the last col"
            return True

        return inner

    @pytest.fixture
    def estimate_roi_memory_consumption(self, checked_config, state_machine):
        """Helper that estimate the memory consumption of the given ROI."""

        def inner(roi):
            checked_config["ROI"] = roi
            height, width = memory_estimation.compute_effective_image_size(
                checked_config, state_machine.margins_img.global_margins
            )
            return memory_estimation.estimate_total_consumption(
                checked_config, height, width, state_machine.margins_disp.global_margins
            )

        return inner

    @pytest.mark.parametrize(
        "segment_mode",
        [
            pytest.param(
                {
                    "enable": True,
                    "memory_per_work": 4000,
                },
                id="Enough memory",
            ),
        ],
    )
    def test_no_segments_when_not_needed(self, checked_config, state_machine):
        """No segment is expected when disabled or with enough memory."""
        result = memory_estimation.segment_image_by_rows(
            checked_config, state_machine.margins_disp.global_margins, state_machine.margins_img.global_margins
        )

        assert len(result) == 0

    @pytest.mark.parametrize(
        ["image_size", "memory_per_work"],
        [
            pytest.param([1001, 1455], 100, id="Small image"),
            pytest.param([1500, 2340], 150, id="Bigger image"),
            pytest.param([1010, 1320], 41, id="Maximum rows per ROI is 1"),
        ],
    )
    def test_enough_memory(
        self,
        checked_config,
        memory_per_work,
        state_machine,
        image_can_be_fully_reconstructed,
        estimate_roi_memory_consumption,
        dataset_disp_map_size,
    ):
        """There is enough memory per work to split image into segments."""
        result = memory_estimation.segment_image_by_rows(
            checked_config, state_machine.margins_disp.global_margins, state_machine.margins_img.global_margins
        )

        assert len(result) >= 2, "There should be at least 2 segments."
        assert all(check_roi_section({"ROI": cast(Dict, e)}).get("ROI") for e in result)
        assert image_can_be_fully_reconstructed(result)
        assert all(
            (estimate_roi_memory_consumption(roi) + dataset_disp_map_size)
            < (1 - memory_estimation.RELATIVE_ESTIMATION_MARGIN) * memory_per_work
            for roi in result
        )

    @pytest.mark.parametrize(
        ["image_size", "memory_per_work"],
        [
            pytest.param([2500, 4500], 170, id="Dataset disp map too big"),
            pytest.param([450, 2200], 34, id="Min ROI too big (width too big)"),
        ],
    )
    def test_raise_error_when_not_enough_memory(
        self,
        checked_config,
        memory_per_work,
        state_machine,
    ):
        """Raise an error when either the initial disparity map size or the minimum ROI (one line) is too large,
        providing an indication of the minimum memory_per_work value required to fit them into memory."""
        with pytest.raises(
            ValueError,
            match=(
                rf"^estimated minimum `memory_per_work` is \d+ MB, but got {memory_per_work} MB. "
                "Consider increasing it, reducing image size or working on ROI."
            ),
        ):
            memory_estimation.segment_image_by_rows(
                checked_config, state_machine.margins_disp.global_margins, state_machine.margins_img.global_margins
            )


class TestSegmentImageByRowsWithRoi(TestSegmentImageByRows):
    """Test segment_image_by_rows with ROI in initial config."""

    @pytest.fixture
    def input_roi(self, image_size):
        """Default ROI taking full image."""
        return {
            "row": {"first": 0, "last": image_size[0] - 1},
            "col": {"first": 0, "last": image_size[1] - 1},
        }

    @pytest.fixture
    def config(self, input_roi, tmp_path, input_config, segment_mode):  # pylint: disable=arguments-differ
        return {
            **input_config,
            "ROI": input_roi,
            "pipeline": {
                "matching_cost": {
                    "matching_cost_method": "mutual_information",
                    "window_size": 5,
                    "step": [1, 1],
                    "subpix": 1,
                },
                "disparity": {
                    "disparity_method": "wta",
                    "invalid_disparity": -9999,
                },
            },
            "segment_mode": segment_mode,
            "output": {
                "path": str(tmp_path),
            },
        }

    @pytest.fixture
    def dataset_disp_map_size(self, checked_config, state_machine):  # pylint: disable=arguments-renamed
        return memory_estimation.estimate_dataset_disp_map_size(
            *memory_estimation.compute_effective_image_size(checked_config, state_machine.margins_img.global_margins),
            checked_config["pipeline"]["matching_cost"]["step"],
            checked_config["pipeline"]["matching_cost"]["float_precision"],
        )

    @pytest.fixture
    def image_can_be_fully_reconstructed(self, input_roi):  # pylint: disable=arguments-renamed
        """Helper that checks that the image can be fully reconstructed from an ROI list."""

        def inner(rois):
            row_sorted = sorted(rois, key=lambda roi: roi["row"]["first"])
            for first_roi, second_roi in zip(row_sorted[:-1], row_sorted[1:]):
                assert second_roi["row"]["first"] == first_roi["row"]["last"] + 1, "ROIs should be continuous"
                assert first_roi["col"] == second_roi["col"], "ROIs should be only on rows."
            # Following asserts are relevant because we previously checked that all ROIs were contiguous:
            assert row_sorted[0]["row"]["first"] == input_roi["row"]["first"]
            assert row_sorted[-1]["row"]["last"] == input_roi["row"]["last"]
            assert row_sorted[0]["col"]["first"] == input_roi["col"]["first"]
            assert row_sorted[-1]["col"]["last"] == input_roi["col"]["last"]
            return True

        return inner

    # Add input_roi
    @pytest.mark.parametrize(
        ["image_size", "input_roi", "memory_per_work"],
        [
            pytest.param(
                [1001, 1455],
                {
                    "row": {"first": 100, "last": 999},
                    "col": {"first": 1000, "last": 1400},
                },
                100,
                id="Small image",
            ),
            pytest.param(
                [4000, 2455],
                {
                    "row": {"first": 100, "last": 1500},
                    "col": {"first": 1000, "last": 1400},
                },
                100,
                id="Bigger image",
            ),
        ],
    )
    def test_enough_memory(
        self,
        checked_config,
        memory_per_work,
        state_machine,
        image_can_be_fully_reconstructed,
        estimate_roi_memory_consumption,
        dataset_disp_map_size,
    ):
        """There is enough memory per work to split image into segments."""
        super().test_enough_memory(
            checked_config,
            memory_per_work,
            state_machine,
            image_can_be_fully_reconstructed,
            estimate_roi_memory_consumption,
            dataset_disp_map_size,
        )

    # Add input_roi
    @pytest.mark.parametrize(
        ["image_size", "input_roi", "memory_per_work"],
        [
            pytest.param(
                [5500, 6500],
                {"row": {"first": 0, "last": 2499}, "col": {"first": 0, "last": 4500}},
                170,
                id="Dataset disp map too big",
            ),
            pytest.param(
                [2000, 2301],
                {"row": {"first": 1000, "last": 1449}, "col": {"first": 100, "last": 2300}},
                34,
                id="Min ROI too big (width too big)",
            ),
        ],
    )
    def test_raise_error_when_not_enough_memory(
        self,
        checked_config,
        memory_per_work,
        state_machine,
    ):
        """Raise an error when either the initial disparity map size or the minimum ROI (one line) is too large,
        providing an indication of the minimum memory_per_work value required to fit them into memory."""
        super().test_raise_error_when_not_enough_memory(checked_config, memory_per_work, state_machine)
