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
Module with global test fixtures.
"""
from typing import List
import sqlite3
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
        self.total_time = items[1]
        self.cpu_usage = items[2]
        self.mem_usage = items[3]


def pytest_addoption(parser):
    parser.addoption("--database", action="store", default=".pymon", required=False)


@pytest.fixture(name="database_path")
def database_path_fixture(request):
    return request.config.getoption("--database")


@pytest.fixture
def output_result_path():
    return "./tests/resource_tests/result"


@pytest.fixture(name="sqlite_select_query")
def sqlite_select_query_fixture():
    return """SELECT ITEM, TOTAL_TIME, CPU_USAGE, MEM_USAGE FROM TEST_METRICS"""


@pytest.fixture()
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
