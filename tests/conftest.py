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

# pylint: disable=redefined-outer-name

import pathlib
import re

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Update marker collections

    1. Adds the marker corresponding to the various directories ending in "_test".
    The various markers created are:
    - unit_tests
    - functional_tests
    - performance_tests
    - resource_tests
    2. Disables pytest_monitor.
    """
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        mark_name = next((part for part in rel_path.parts if part.endswith("_tests")), "")
        if mark_name:
            mark = getattr(pytest.mark, mark_name)
            item.add_marker(mark)
            item.add_marker(pytest.mark.monitor_skip_test)


def pytest_html_results_table_header(cells):
    """
    Add columns to html reports:

    1. Category : with values {'TU', 'TF', 'TP', 'TR'}
    2. Function tested : basename of python test file
    """
    cells.insert(1, "<th>Category</th>")
    cells.insert(2, "<th>Function tested</th>")


def pytest_html_results_table_row(report, cells):
    """
    Complete columns to html reports with regex pattern :

    "tests/<CATEGORY>_tests/.../tests_<FUNCTION>.py::tests"

    1. CATEGORY : with values {'TU', 'TF', 'TP', 'TR'}
    2. FUNCTION : basename of python test file
    """
    type_dict = {"unit": "TU", "functional": "TF", "resource": "TR", "performance": "TP"}
    pattern = r"tests/(?P<type>\w+)_tests.*test_(?P<function>\w+)\.py"
    match = re.match(pattern, report.nodeid)
    cells.insert(1, f"<td>{type_dict[match.groupdict()['type']]}</td>")
    cells.insert(2, f"<td>{match.groupdict()['function']}</td>")


@pytest.fixture()
def classic_config():
    return "./tests/data/json_conf_files/classic_cfg.json"


@pytest.fixture()
def left_img_path():
    return "./tests/data/images/cones/monoband/left.png"


@pytest.fixture()
def right_img_path():
    return "./tests/data/images/cones/monoband/right.png"


@pytest.fixture()
def left_rgb_path():
    return "./tests/data/images/cones/multibands/left.tif"


@pytest.fixture()
def right_rgb_path():
    return "./tests/data/images/cones/multibands/right.tif"


@pytest.fixture
def correct_input_cfg(left_img_path, right_img_path):
    return {
        "input": {
            "left": {
                "img": left_img_path,
                "nodata": "NaN",
            },
            "right": {
                "img": right_img_path,
            },
            "col_disparity": [-2, 2],
            "row_disparity": [-2, 2],
        }
    }


@pytest.fixture
def correct_multiband_input_cfg(left_rgb_path, right_rgb_path):
    return {
        "input": {
            "left": {
                "img": left_rgb_path,
                "nodata": "NaN",
            },
            "right": {
                "img": right_rgb_path,
            },
            "col_disparity": [-2, 2],
            "row_disparity": [-2, 2],
        }
    }
