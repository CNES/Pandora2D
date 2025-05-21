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
from pandora.margins import Margins
from pandora2d import memory_estimation
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing
from pandora2d.check_configuration import check_conf
from pandora2d.state_machine import Pandora2DMachine


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
        ["row_disparity", "col_disparity", "expected"],
        [
            pytest.param(
                {"init": 0, "range": 2},
                {"init": 0, "range": 3},
                (5, 7),
                id="Centered disparities",
            ),
            pytest.param(
                {"init": 3, "range": 1},
                {"init": 0, "range": 3},
                (3, 7),
                id="Positive row disparity",
            ),
            pytest.param(
                {"init": 0, "range": 2},
                {"init": -5, "range": 4},
                (5, 9),
                id="Negative col disparity",
            ),
        ],
    )
    def test_get_nb_disp(self, row_disparity, col_disparity, expected):
        """
        Test the get_nb_disp method
        """

        assert memory_estimation.get_nb_disp(row_disparity, col_disparity) == expected

    @pytest.mark.parametrize(
        ["row_disparity", "col_disparity", "expected"],
        [
            pytest.param(
                "correct_grid",
                "second_correct_grid",
                (14, 37),
                id="Grid disparities",
            ),
        ],
    )
    def test_get_nb_disp_with_grid(self, row_disparity, col_disparity, expected, request):
        """
        Test the get_nb_disp method with grid disparities
        """

        assert (
            memory_estimation.get_nb_disp(
                request.getfixturevalue(row_disparity), request.getfixturevalue(col_disparity)
            )
            == expected
        )

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

        assert pytest.approx(memory_estimation.input_size(height, width, data_vars), abs=1e-3) == expected

    def test_memory_input(self, correct_input_cfg):
        """
        Test that the value returned by input_size corresponds to the memory occupied by the image datasets.
        """

        # Memory computed by input_size method
        height, width = memory_estimation.get_img_size(correct_input_cfg["input"]["left"]["img"])
        data_variables = memory_estimation.IMG_DATA_VAR

        # We have a factor of x2 for left and right images
        memory_computed = 2 * memory_estimation.input_size(height, width, data_variables)

        # Memory consumed when creating the two images datasets
        tracemalloc.start()
        image_datasets = create_datasets_from_inputs(correct_input_cfg["input"])
        current, _ = tracemalloc.get_traced_memory()
        current_mb = current / memory_estimation.BYTE_TO_MB
        tracemalloc.stop()

        # Check that the estimated image dataset memory corresponds to the measured memory within 10%.
        assert memory_computed == pytest.approx(current_mb, rel=0.10)
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
        memory_computed = 2 * memory_estimation.input_size(height, width, data_variables)

        # Memory consumed when creating the two images datasets
        tracemalloc.start()
        image_datasets = create_datasets_from_inputs(cfg["input"], roi)
        current, _ = tracemalloc.get_traced_memory()
        current_mb = current / memory_estimation.BYTE_TO_MB
        tracemalloc.stop()

        # Check that the estimated image dataset memory corresponds to the measured memory within 25%.
        # Estimated dataset size is 0.37 and measured dataset size is 0.39.
        # Total measured memory is 0.47.
        assert memory_computed == pytest.approx(current_mb, rel=0.25)
        # Check that the estimated image dataset memory corresponds to the result of
        # image_dataset.left.nbytes +  image_dataset.right.nbytes
        image_dataset_nbytes = (image_datasets.left.nbytes + image_datasets.right.nbytes) / memory_estimation.BYTE_TO_MB
        assert memory_computed == pytest.approx(image_dataset_nbytes, rel=0.05)
