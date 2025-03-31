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
This module is used to check the resources used for the tests in this directory.
"""

# pylint: disable=protected-access

import sqlite3
from typing import List

import pytest


class Metrics:
    """
    Metrics Class
    """

    # pylint:disable=too-few-public-methods

    _TOTAL_TIME_MAX = 120  # second
    _CPU_USAGE_MAX = 50  # percent
    _MEM_USAGE_MAX = 1024  # megabyte

    def __init__(self, items: List) -> None:
        self.test_name = items[0]
        self.test_variant = items[1]
        self.total_time = items[2]
        self.cpu_usage = items[3]
        self.mem_usage = items[4]


def read_sqlite_table(database_path, sqlite_select_query):
    """
    Read sqlite table from pytest-monitoring
    """
    data = []
    try:
        sqlite_connection = sqlite3.connect(database_path)
        cursor = sqlite_connection.cursor()

        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()
        data = [Metrics(record) for record in records]
        sqlite_connection.close()
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    return data


# Define the pytest_generate_tests hook to generate test cases
def pytest_generate_tests(metafunc):
    """Generate list of tests from pytest-monitoring database."""
    query = "SELECT ITEM, ITEM_VARIANT, TOTAL_TIME, CPU_USAGE, MEM_USAGE FROM TEST_METRICS"
    marks = [mark.name for mark in metafunc.cls.pytestmark]
    if "metrics" in marks:
        metrics = read_sqlite_table(metafunc.config.option.database, query)
        if metrics:
            # Generate test cases based on the metrics list
            metafunc.parametrize("metric", metrics, ids=lambda x: x.test_variant)


@pytest.mark.metrics
@pytest.mark.monitor_skip_test
class TestResource:
    """
    Test all tests are ok for CPU/MEM and time rule
    """

    def test_total_time(self, metric):
        """
        Verify the time metrics for the test
        """
        assert (
            metric.total_time < metric._TOTAL_TIME_MAX
        ), f"Test {metric.test_variant} does not respect max time : {metric._TOTAL_TIME_MAX} (seconds)"

    def test_cpu_usage(self, metric):
        """
        Verify the cpu metrics for the test
        """
        assert (
            metric.cpu_usage < metric._CPU_USAGE_MAX
        ), f"Test {metric.test_variant} does not cpu usage max : {metric._CPU_USAGE_MAX} (%)"

    def test_mem_usage(self, metric):
        """
        Verify the memory metrics for the test
        """
        assert (
            metric.mem_usage < metric._MEM_USAGE_MAX
        ), f"Test {metric.test_variant} does not respect memory usage max : {metric._MEM_USAGE_MAX} (megabyte)"
