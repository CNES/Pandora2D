#!/usr/bin/env python
#
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
Tests
"""

from contextlib import nullcontext as does_not_raise

import sys
import pytest


class TestExample:
    """
    Examples
    """

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, 2),
            pytest.param(1, 0, marks=pytest.mark.xfail),
            pytest.param(1, 3, marks=pytest.mark.xfail(reason="some bug")),
            (2, 3),
            (3, 4),
            (4, 5),
            pytest.param(10, 11, marks=pytest.mark.skipif(sys.version_info >= (3, 0), reason="py2k")),
        ],
    )
    def test_increment(self, n, expected):
        """
        Test with xfail and skipif
        """
        assert n + 1 == expected

    @pytest.mark.parametrize(
        "example_input,expectation",
        [
            (3, does_not_raise()),
            (2, does_not_raise()),
            (1, does_not_raise()),
            (0, pytest.raises(ZeroDivisionError)),
        ],
    )
    def test_division(self, example_input, expectation):
        """Test how much I know division."""
        with expectation:
            assert (6 / example_input) is not None
