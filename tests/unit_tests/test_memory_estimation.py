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

import tracemalloc

import pytest

from pandora2d import memory_estimation
from pandora2d.check_configuration import check_conf
from pandora2d.criteria import get_criteria_dataarray
from pandora2d.img_tools import (
    create_datasets_from_inputs,
    get_roi_processing,
    shift_subpix_img_2d,
)
from pandora2d.margins import Margins
from pandora2d.state_machine import Pandora2DMachine


class MemoryTracer:
    """
    Measure consumed memory in bytes.
    """

    def __init__(self, unit_factor=1):
        self.unit_factor = unit_factor
        self._current = 0
        self._peak = 0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._current, self._peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    @property
    def current(self):
        return self._current / self.unit_factor

    @property
    def peak(self):
        return self._peak / self.unit_factor


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
        ["disparity", "expected"],
        [
            pytest.param(
                {"init": 0, "range": 2},
                5,
                id="Centered disparities",
            ),
            pytest.param(
                {"init": 3, "range": 1},
                3,
                id="Positive disparity",
            ),
            pytest.param(
                {"init": -5, "range": 4},
                9,
                id="Negative disparity",
            ),
        ],
    )
    def test_get_nb_disp(self, disparity, expected):
        """
        Test the get_nb_disp method
        """

        assert memory_estimation.get_nb_disp(disparity) == expected

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
                id="egative col disparity",
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

    def test_memory_input(self, correct_input_cfg):
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

    def test_memory_input_with_roi(self, correct_input_cfg, correct_roi_sensor, correct_pipeline):
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
    def test_memory_cost_volumes(self, user_cfg_cv_memory, matching_cost_object):
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


class TestShiftedRightImages:
    """Test memory consumption of shifted right images."""

    @pytest.fixture()
    def right_image(self, correct_input_cfg):
        return create_datasets_from_inputs(correct_input_cfg["input"]).right

    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test(self, right_image, subpix):
        """Test memory consumption of shifted right images."""

        with MemoryTracer(memory_estimation.BYTE_TO_MB) as memory_tracer:
            images = shift_subpix_img_2d(right_image, subpix)
        # We exclude the first image from the count as it is excluded in estimate_shifted_right_images_size
        images_size = sum(image.nbytes for image in images[1:]) / memory_estimation.BYTE_TO_MB

        result = memory_estimation.estimate_shifted_right_images_size(
            right_image.dims["row"], right_image.dims["col"], subpix
        )

        assert result == pytest.approx(images_size, rel=0.05)
        # When subpix = 1, we approximate with absolute tolerance since we expect a value close to 0,
        # making relative tolerance irrelevant in this case.
        assert result == pytest.approx(memory_tracer.current, rel=0.95, abs=1e-2)
