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
#

"""
Test automatic ZNCC implementation selection.
"""

import numpy as np
import pytest
from pytest_mock import MockerFixture

from pandora2d import matching_cost
from pandora2d.matching_cost.correlation import get_roi_area, select_zncc_optim_method

# Areas of the ROI used in the ZNCC benchmark the selection model was fitted on.
SMALL_ROI_AREA = 512 * 512
LARGE_ROI_AREA = 768 * 768
HUGE_ROI_AREA = 1024 * 1024


@pytest.mark.parametrize(
    ("window_size", "step", "roi_area", "expected_method"),
    [
        pytest.param(5, [1, 1], 25, "zncc-optim-1", id="dense_sampling"),
        pytest.param(5, [5, 5], 25, "zncc-optim-2", id="sparse_sampling"),
        pytest.param(5, [2, 1], 25, "zncc-optim-1", id="asymmetric_step"),
        pytest.param(3, [1, 1], 25, "zncc-optim-1", id="small_window_dense_sampling"),
        pytest.param(65, [32, 32], HUGE_ROI_AREA, "zncc-optim-1", id="large_window_compensates_step"),
        pytest.param(65, [60, 60], HUGE_ROI_AREA, "zncc-optim-2", id="step_outweighs_large_window"),
        # The next four cases only differ by the ROI area: it is what makes the selection switch.
        pytest.param(5, [1, 1], SMALL_ROI_AREA, "zncc-optim-1", id="small_window_small_roi"),
        pytest.param(5, [1, 1], LARGE_ROI_AREA, "zncc-optim-2", id="small_window_large_roi"),
        pytest.param(17, [8, 8], SMALL_ROI_AREA, "zncc-optim-1", id="medium_window_small_roi"),
        pytest.param(17, [8, 8], HUGE_ROI_AREA, "zncc-optim-2", id="medium_window_huge_roi"),
    ],
)
def test_select_zncc_optim_method(window_size, step, roi_area, expected_method):
    """
    Description : Test ZNCC implementation selection from window size, step and ROI area.
    """
    assert select_zncc_optim_method(window_size, step, roi_area) == expected_method


@pytest.mark.parametrize(
    ("roi", "expected_area"),
    [
        pytest.param(None, 25, id="no_roi_falls_back_to_image_area"),
        pytest.param({"row": {"first": 0, "last": 2}, "col": {"first": 1, "last": 4}}, 12, id="rectangular_roi"),
        pytest.param(
            {"row": {"first": 128, "last": 895}, "col": {"first": 128, "last": 895}},
            LARGE_ROI_AREA,
            id="square_roi",
        ),
    ],
)
def test_get_roi_area(roi, expected_area, make_dataset):
    """
    Description : Test that the ROI area is read from the configuration, or from the left image without ROI.
    """
    left_dataset = make_dataset(np.ones((5, 5), dtype=np.float32))
    cfg = {} if roi is None else {"ROI": roi}

    assert get_roi_area(cfg, left_dataset) == expected_area


@pytest.mark.parametrize(
    ("matching_cost_method", "window_size", "step", "expected_cpp_method"),
    [
        pytest.param("zncc", 5, [1, 1], "zncc-optim-1", id="auto_zncc_dense_sampling"),
        pytest.param("zncc", 5, [5, 5], "zncc-optim-2", id="auto_zncc_sparse_sampling"),
        pytest.param("zncc", 5, [2, 1], "zncc-optim-1", id="auto_zncc_asymmetric_step"),
        pytest.param("zncc-optim-1", 5, [5, 5], "zncc-optim-1", id="forced_optim_1"),
        pytest.param("zncc-optim-2", 5, [1, 1], "zncc-optim-2", id="forced_optim_2"),
        pytest.param("mutual_information", 5, [1, 1], "mutual_information", id="mutual_information"),
    ],
)
def test_compute_cost_volumes_passes_resolved_cpp_method(
    matching_cost_method,
    window_size,
    step,
    expected_cpp_method,
    make_dataset,
    matching_cost_config,
    mocker: MockerFixture,
):
    """
    Description : Test that compute_cost_volumes_cpp receives the resolved correlation method.
    """
    mock_cpp = mocker.patch("pandora2d.matching_cost.correlation.matching_cost_bind.compute_cost_volumes_cpp_float")

    data = np.ones((5, 5), dtype=np.float32)
    left_dataset = make_dataset(data)
    right_dataset = make_dataset(data)

    correlation_matcher = matching_cost.CorrelationMethods(matching_cost_config)
    correlation_matcher.allocate(left_dataset, right_dataset, matching_cost_config)
    correlation_matcher.compute_cost_volumes(left_dataset, right_dataset)

    assert mock_cpp.call_args.args[-1] == expected_cpp_method


@pytest.mark.parametrize("matching_cost_method", ["zncc"])
@pytest.mark.parametrize("window_size", [5])
@pytest.mark.parametrize("step", [[1, 1]])
@pytest.mark.parametrize(
    ("roi", "expected_cpp_method"),
    [
        pytest.param(None, "zncc-optim-1", id="without_roi"),
        pytest.param(
            {"row": {"first": 128, "last": 895}, "col": {"first": 128, "last": 895}, "margins": [0, 0, 0, 0]},
            "zncc-optim-2",
            id="with_large_roi",
        ),
    ],
)
def test_roi_area_is_taken_into_account(
    roi,
    expected_cpp_method,
    make_dataset,
    matching_cost_config,
    mocker: MockerFixture,
):
    """
    Description : Test that the ROI given at allocation switches the auto-selected implementation
    for identical window size and step.
    """
    mock_cpp = mocker.patch("pandora2d.matching_cost.correlation.matching_cost_bind.compute_cost_volumes_cpp_float")

    data = np.ones((5, 5), dtype=np.float32)
    left_dataset = make_dataset(data)
    right_dataset = make_dataset(data)

    # The C++ call is mocked, so the cost volumes are never filled: only the ROI area read from the
    # configuration matters here, not the fact that it is wider than the test images.
    cfg = dict(matching_cost_config) if roi is None else {**matching_cost_config, "ROI": roi}

    correlation_matcher = matching_cost.CorrelationMethods(matching_cost_config)
    correlation_matcher.allocate(left_dataset, right_dataset, cfg)
    correlation_matcher.compute_cost_volumes(left_dataset, right_dataset)

    assert mock_cpp.call_args.args[-1] == expected_cpp_method


def test_unsupported_correlation_method_raises_error(mocker: MockerFixture):
    """
    Description : Test that _resolve_cpp_correlation_method raises ValueError for unsupported methods.
    """
    cfg = {
        "matching_cost_method": "mutual_information",
        "window_size": 5,
        "step": [1, 1],
        "subpix": 1,
        "float_precision": "float32",
    }

    correlation_matcher = matching_cost.CorrelationMethods(cfg)

    # Simulate receiving an unsupported method by patching the internal _method attribute
    # This tests the defensive programming layer in _resolve_cpp_correlation_method
    correlation_matcher._method = "invalid_method"

    with pytest.raises(ValueError, match="Unsupported correlation method"):
        correlation_matcher._resolve_cpp_correlation_method()
