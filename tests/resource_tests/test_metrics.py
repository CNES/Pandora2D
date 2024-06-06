# pylint: disable=protected-access
#!/usr/bin/env python
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module is used to check the resources used for the tests in this directory.
"""
import pytest


@pytest.mark.metrics
@pytest.mark.monitor_skip_test
class TestResource:
    """
    Test all tests are ok for CPU/MEM and time rule
    """

    def test_total_time(self, read_sqlite_table):
        """
        Verify the time metrics for the test
        """
        for metric in read_sqlite_table:
            assert (
                metric.total_time < metric._TOTAL_TIME_MAX
            ), f"Test {metric.test_name} does not respect max time : {metric._TOTAL_TIME_MAX} (seconds)"

    def test_cpu_usage(self, read_sqlite_table):
        """
        Verify the cpu metrics for the test
        """
        for metric in read_sqlite_table:
            assert (
                metric.cpu_usage < metric._CPU_USAGE_MAX
            ), f"Test {metric.test_name} does not cpu usage max : {metric._CPU_USAGE_MAX} (%)"

    def test_mem_usage(self, read_sqlite_table):
        """
        Verify the memory metrics for the test
        """
        for metric in read_sqlite_table:
            assert (
                metric.mem_usage < metric._MEM_USAGE_MAX
            ), f"Test {metric.test_name} does not respect memory usage max : {metric._MEM_USAGE_MAX} (megabyte)"
