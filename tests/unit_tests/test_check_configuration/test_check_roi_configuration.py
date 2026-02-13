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
Test check input step configuration
"""

import pytest

from pandora2d.check_configuration import check_roi_coherence, check_roi_section


class TestCheckRoiSection:
    """
    Description : Test check_roi_section.
    Requirement : EX_ROI_04
    """

    def test_nominal_case(self, correct_roi_sensor) -> None:
        """
        Test function for checking user ROI section
        """
        # with a correct ROI check_roi_section should return nothing
        check_roi_section(correct_roi_sensor)

    def test_dimension_lt_0_raises_exception(self, false_roi_sensor_negative):
        """
        Description : Raises an exception if the ROI dimensions are lower than 0
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(BaseException):
            check_roi_section(false_roi_sensor_negative)

    def test_first_dimension_gt_last_dimension_raises_exception(self, false_roi_sensor_first_superior_to_last):
        """
        Description : Test if the first dimension of the ROI is greater than the last one
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(BaseException):
            check_roi_section(false_roi_sensor_first_superior_to_last)

    @pytest.mark.parametrize(
        "roi_section",
        [
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 10},
                        "row": {"first": 10, "last": 100},
                    },
                },
                id="Only col",
            ),
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 100},
                        "row": {"first": 10, "last": 10},
                    },
                },
                id="Only row",
            ),
            pytest.param(
                {
                    "ROI": {
                        "col": {"first": 10, "last": 10},
                        "row": {"first": 10, "last": 10},
                    },
                },
                id="Both row and col",
            ),
        ],
    )
    def test_one_pixel_roi_is_valid(self, roi_section):
        """Should not raise error."""
        check_roi_section(roi_section)


class TestCheckRoiCoherence:
    """
    Description : Test check_roi_coherence.
    Requirement : EX_ROI_04
    """

    def test_first_lt_last_is_ok(self, correct_roi_sensor) -> None:
        check_roi_coherence(correct_roi_sensor["ROI"]["col"])

    def test_first_gt_last_raises_error(self, false_roi_sensor_first_superior_to_last):
        """
        Description : Test if 'first' is greater than 'last' in ROI
        Data :
        Requirement : EX_CONF_08
        """
        with pytest.raises(ValueError) as exc_info:
            check_roi_coherence(false_roi_sensor_first_superior_to_last["ROI"]["col"])
        assert str(exc_info.value) == '"first" should be lower than "last" in sensor ROI'
