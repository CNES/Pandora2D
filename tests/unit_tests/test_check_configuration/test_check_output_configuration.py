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
Test check output step configuration
"""

from copy import deepcopy

import pytest
from json_checker import DictCheckerError

from pandora2d.check_configuration import check_output_section


class TestCheckOutputSection:
    """Test check_output_section"""

    @pytest.fixture
    def output_configuration(self):
        """Default output configuration"""
        return {"output": {"path": "/home/me/out"}}

    def test_path_is_mandatory(self):
        """Check output key is mandatory"""
        with pytest.raises(DictCheckerError, match="path"):
            check_output_section({"output": {}})

    @pytest.mark.parametrize("format_", ["tiff"])
    def test_accept_optional_format(self, format_, output_configuration):
        """Check optional format parameter"""
        config = deepcopy(output_configuration)
        config["output"]["format"] = format_
        check_output_section(config)

    @pytest.mark.parametrize("format_", ["unknown"])
    def test_fails_with_bad_format(self, format_, output_configuration):
        """Check wrong format parameter"""
        config = deepcopy(output_configuration)
        config["output"]["format"] = format_
        with pytest.raises(DictCheckerError, match="format"):
            check_output_section(config)

    @pytest.mark.parametrize(
        "deformation_grid",
        [{"init_pixel_conv_grid": [0, 0]}, {"init_pixel_conv_grid": [0.5, 0.5]}],
    )
    def test_accept_optional_deformation_grid(self, deformation_grid, output_configuration):
        """
        Check that optional deformation_grid key is accepted in output configuration
        """
        config = deepcopy(output_configuration)
        config["output"]["deformation_grid"] = deformation_grid
        check_output_section(config)

    @pytest.mark.parametrize(
        "deformation_grid",
        [
            {"init_pixel_conv_grid": [0, 0.5]},
            {"init_pixel_conv_grid": [0.5, 0.0]},
            {"init_pixel_conv_grid": "wrong_type"},
            {},
            {"wrong_key": [0, 0]},
        ],
    )
    def test_fails_with_wrong_deformation_grid(self, deformation_grid, output_configuration):
        """
        Check that check_output_section fails when using a wrong deformation_grid type
        """
        config = deepcopy(output_configuration)
        config["output"]["deformation_grid"] = deformation_grid
        with pytest.raises(DictCheckerError, match="deformation_grid"):
            check_output_section(config)
