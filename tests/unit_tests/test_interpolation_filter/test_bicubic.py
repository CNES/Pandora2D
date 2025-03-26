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
"""Test the bicubic filter module."""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from pandora.margins import Margins

import pandora2d.interpolation_filter


@pytest.fixture()
def filter_instance():
    return pandora2d.interpolation_filter.AbstractFilter(  # pylint: disable=abstract-class-instantiated
        cfg={"method": "bicubic_python"},
    )  # type: ignore[abstract]


@pytest.fixture()
def filter_instance_cpp():
    return pandora2d.interpolation_filter.AbstractFilter(  # pylint: disable=abstract-class-instantiated
        cfg={"method": "bicubic"},
    )  # type: ignore[abstract]


@pytest.fixture()
def make_filter_instance(request):
    return request.getfixturevalue(request.param["filter"])


@pytest.mark.parametrize(
    ["make_filter_instance", "gt_instance"],
    [
        pytest.param(
            {"filter": "filter_instance"},
            pandora2d.interpolation_filter.bicubic.BicubicPython,
            id="Bicubic python",
        ),
        pytest.param(
            {"filter": "filter_instance_cpp"},
            pandora2d.interpolation_filter.bicubic_cpp.Bicubic,
            id="Bicubic cpp",
        ),
    ],
    indirect=["make_filter_instance"],
)
def test_factory(make_filter_instance, gt_instance):
    """
    Test instances of Bicubic filters
    """
    assert isinstance(make_filter_instance, gt_instance)


@pytest.mark.parametrize(
    ["make_filter_instance"],
    [
        pytest.param(
            {"filter": "filter_instance"},
            id="Bicubic python",
        ),
        pytest.param(
            {"filter": "filter_instance_cpp"},
            id="Bicubic cpp",
        ),
    ],
    indirect=["make_filter_instance"],
)
def test_margins(make_filter_instance):
    """
    Test margins values
    """
    assert make_filter_instance.margins == Margins(1, 1, 2, 2)


@pytest.mark.parametrize(
    ["make_filter_instance"],
    [
        pytest.param(
            {"filter": "filter_instance"},
            id="Bicubic python",
        ),
        pytest.param(
            {"filter": "filter_instance_cpp"},
            id="Bicubic cpp",
        ),
    ],
    indirect=["make_filter_instance"],
)
@pytest.mark.parametrize(
    ["coeff", "expected"],
    [
        (0, [0, 1, 0, 0]),
        (0.5, [-0.0625, 0.5625, 0.5625, -0.0625]),
        (0.25, [-0.0703125, 0.8671875, 0.2265625, -0.0234375]),
    ],
)
def test_get_coeffs_computation(make_filter_instance, coeff, expected):
    """Test result of get_coeff computation."""
    result = make_filter_instance.get_coeffs(coeff)
    np.testing.assert_array_equal(result, expected)
