#!/usr/bin/env python
#
# Copyright (c) 2025 CS GROUP France
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
Test Matching cost class
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import json_checker
from pandora2d import estimation


@pytest.fixture()
def full_configuration():
    return {
        "estimation_method": "phase_cross_correlation",
        "range_col": 5,
        "range_row": 5,
        "sample_factor": 10,
    }


@pytest.fixture()
def estimation_class(full_configuration):
    """Build estimation object."""
    estimation_class = estimation.AbstractEstimation(full_configuration)  # type: ignore[abstract]
    return estimation_class


@pytest.mark.parametrize(
    ["estimation_method", "range_col", "range_row", "sample_factor", "error"],
    [
        pytest.param("another_method", 5, 5, 10, KeyError, id="Wrong method"),
        pytest.param(
            "phase_cross_correlation", -1, 5, 10, json_checker.core.exceptions.DictCheckerError, id="negative range_col"
        ),
        pytest.param(
            "phase_cross_correlation", "5", 5, 10, json_checker.core.exceptions.DictCheckerError, id="string range_col"
        ),
        pytest.param(
            "phase_cross_correlation", 5, 0, 10, json_checker.core.exceptions.DictCheckerError, id="0 as range_row"
        ),
        pytest.param(
            "phase_cross_correlation", 5, [2, 3], 10, json_checker.core.exceptions.DictCheckerError, id="list range_row"
        ),
        pytest.param(
            "phase_cross_correlation", 5, 5, 0, json_checker.core.exceptions.DictCheckerError, id="0 for sample factor"
        ),
        pytest.param(
            "phase_cross_correlation",
            5,
            5,
            15,
            json_checker.core.exceptions.DictCheckerError,
            id="not a multiple of 10 sample factor",
        ),
        pytest.param(
            "phase_cross_correlation", 6, 5, 1, json_checker.core.exceptions.DictCheckerError, id="even range_row"
        ),
        pytest.param(
            "phase_cross_correlation", 5, 6, 1, json_checker.core.exceptions.DictCheckerError, id="even range_col"
        ),
    ],
)
def test_false_check_conf(estimation_method, range_col, range_row, sample_factor, error):
    """
    Description : test check_conf of estimation with wrongs pipelines
    Data :
    Requirement : EX_CONF_08
    """

    with pytest.raises(error):
        estimation.AbstractEstimation(
            {
                "estimation_method": estimation_method,
                "range_col": range_col,
                "range_row": range_row,
                "sample_factor": sample_factor,
            }
        )  # type: ignore[abstract]


def test_check_conf():
    """
    test check_conf of estimation with a correct pipeline
    """
    estimation.AbstractEstimation(
        {
            "estimation_method": "phase_cross_correlation",
            "range_col": 5,
            "range_row": 5,
            "sample_factor": 10,
        }
    )  # type: ignore[abstract]


@pytest.mark.parametrize(
    ["parameter", "expected_value"],
    [
        ["range_col", 5],
        ["range_row", 5],
        ["sample_factor", 1],
    ],
)
def test_default_parameters_values(full_configuration, parameter, expected_value):
    """
    Description : Test default values are the expected ones.
    Data :
    Requirement : EX_CONF_04
    """
    del full_configuration[parameter]

    result = estimation.AbstractEstimation(full_configuration)  # type: ignore[abstract]

    assert result.cfg[parameter] == expected_value


def test_update_cfg_with_estimation(estimation_class):
    """
    test update_cfg_with_estimation function
    """

    gt_cfg = {
        "input": {"col_disparity": {"init": 1, "range": 2}, "row_disparity": {"init": 1, "range": 2}},
        "pipeline": {"estimation": {"estimated_shifts": [-0.5, 1.3], "error": [1.0], "phase_diff": [1.0]}},
    }

    cfg = estimation_class.update_cfg_with_estimation(
        {"input": {}, "pipeline": {"estimation": {}}},
        {"init": 1, "range": 2},
        {"init": 1, "range": 2},
        -np.array([0.5, -1.3]),
        {"error": np.array([1.0]), "phase_diff": np.array([1.0])},
    )

    assert gt_cfg == cfg


def test_estimation_computation(left_stereo_object, right_stereo_object, estimation_class):
    """
    test estimation_computation with phase_cross_correlation
    """

    left, right = left_stereo_object, right_stereo_object

    estimation_ = estimation_class

    row_disparity, col_disparity, shifts, extra_dict = estimation_.compute_estimation(left, right)

    assert col_disparity == {"init": 0, "range": 5}
    assert row_disparity == {"init": -1, "range": 5}
    assert np.array_equal(shifts, [-0.8, 0])
    assert extra_dict["error"] == 0.9999999999855407
    assert extra_dict["phase_diff"] == "1.06382330e-18"
