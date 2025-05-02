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
Module with global test fixtures.
"""

# pylint: disable=redefined-outer-name
import pathlib
import re

import json
import numpy as np
import pytest
import rasterio
import xarray as xr
from pandora.common import write_data_array

import pandora2d


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
    3. Requirement : validating the Pandora2D tool, string with EX_*
    """
    cells.insert(1, "<th>Category</th>")
    cells.insert(2, "<th>Function tested</th>")
    cells.insert(3, "<th>Requirement</th>")


def pytest_html_results_table_row(report, cells):
    """
    Complete columns to html reports with regex pattern :

    "tests/<CATEGORY>_tests/.../tests_<FUNCTION>.py::tests"

    1. CATEGORY : with values {'TU', 'TF', 'TP', 'TR'}
    2. FUNCTION : basename of python test file
    3. REQUIREMENT : with values EX_*
    """
    type_dict = {"unit": "TU", "functional": "TF", "resource": "TR", "performance": "TP"}
    pattern = r"tests/(?P<type>\w+)_tests.*test_(?P<function>\w+)\.py"
    match = re.match(pattern, report.nodeid)
    cells.insert(1, f"<td>{type_dict[match.groupdict()['type']]}</td>")
    cells.insert(2, f"<td>{match.groupdict()['function']}</td>")
    cells.insert(3, f"<td>{'<br>'.join(report.requirement)}</td>")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # pylint: disable=unused-argument
    """
    Parse test docstrings and retrieve strings in EX_*.
    """
    outcome = yield
    report = outcome.get_result()
    pattern = r"(EX_\w*)"
    report.requirement = re.findall(pattern, str(item.function.__doc__))


@pytest.fixture(scope="session")
def classic_config(root_dir):
    return str(root_dir / "tests/data/json_conf_files/classic_cfg.json")


@pytest.fixture(scope="session")
def root_dir(request):
    return request.session.path


@pytest.fixture(scope="session")
def left_img_path(root_dir):
    return str(root_dir / "tests/data/images/cones/monoband/left.png")


@pytest.fixture(scope="session")
def right_img_path(root_dir):
    return str(root_dir / "tests/data/images/cones/monoband/right.png")


@pytest.fixture(scope="session")
def left_rgb_path(root_dir):
    return str(root_dir / "tests/data/images/cones/multibands/left.tif")


@pytest.fixture(scope="session")
def right_rgb_path(root_dir):
    return str(root_dir / "tests/data/images/cones/multibands/right.tif")


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
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 2},
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
            "col_disparity": {"init": 1, "range": 2},
            "row_disparity": {"init": 1, "range": 2},
        }
    }


@pytest.fixture
def mask_path(left_img_path, tmp_path):
    """Create a mask and save it in tmp"""

    with rasterio.open(left_img_path) as src:
        width = src.width
        height = src.height

    mask = xr.DataArray(data=0, dims=["height", "width"], coords={"height": range(height), "width": range(width)})
    mask[0 : int(height / 2), 0 : int(width / 2)] = 1

    path = tmp_path / "mask_left.tif"

    write_data_array(
        data_array=mask,
        filename=str(path),
    )

    return path


@pytest.fixture
def correct_input_with_left_mask(left_img_path, right_img_path, mask_path):
    return {
        "input": {
            "left": {"img": left_img_path, "nodata": -9999, "mask": str(mask_path)},
            "right": {
                "img": right_img_path,
            },
            "col_disparity": {"init": 0, "range": 2},
            "row_disparity": {"init": 0, "range": 2},
        }
    }


@pytest.fixture
def correct_input_with_right_mask(left_img_path, right_img_path, mask_path):
    return {
        "input": {
            "left": {
                "img": left_img_path,
                "nodata": -9999,
            },
            "right": {"img": right_img_path, "mask": str(mask_path)},
            "col_disparity": {"init": 0, "range": 2},
            "row_disparity": {"init": 0, "range": 2},
        }
    }


@pytest.fixture
def correct_input_with_left_right_mask(left_img_path, right_img_path, mask_path):
    return {
        "input": {
            "left": {"img": left_img_path, "nodata": -9999, "mask": str(mask_path)},
            "right": {"img": right_img_path, "mask": str(mask_path)},
            "col_disparity": {"init": 0, "range": 2},
            "row_disparity": {"init": 0, "range": 2},
        }
    }


@pytest.fixture()
def random_seed():
    """
    Seed generated with:

    >>> import secrets
    >>> secrets.randbits(128)
    """
    return 160187526967402499820683775418299155210


@pytest.fixture()
def random_generator(random_seed):
    return np.random.default_rng(random_seed)


@pytest.fixture()
def run_pipeline(tmp_path):
    """Fixture that returns a function to run a pipeline and which returns the output directory path."""

    def run(configuration):
        config_path = tmp_path / "config.json"
        with config_path.open("w", encoding="utf-8") as file_:
            json.dump(configuration, file_, indent=2)

        pandora2d.main(config_path, verbose=False)
        return tmp_path

    return run


@pytest.fixture()
def constant_initial_disparity():
    """
    Create a correct disparity dictionary
    with constant initial disparity
    """
    return {"init": 1, "range": 3}


@pytest.fixture()
def second_constant_initial_disparity():
    """
    Create a correct disparity dictionary
    with constant initial disparity
    """
    return {"init": 0, "range": 2}


@pytest.fixture()
def make_input_cfg(left_img_path, right_img_path, request):
    """Get input configuration with given disparities"""

    input_cfg = {
        "left": {
            "img": left_img_path,
            "nodata": -9999,
        },
        "right": {"img": right_img_path, "nodata": -9999},
        "col_disparity": request.getfixturevalue(request.param["col_disparity"]),
        "row_disparity": request.getfixturevalue(request.param["row_disparity"]),
    }

    return input_cfg


@pytest.fixture
def left_img_shape(left_img_path):
    """
    Get shape of left image stored in left_img_path fixture
    """

    with rasterio.open(left_img_path) as src:
        width = src.width
        height = src.height

    return (height, width)


@pytest.fixture
def create_disparity_grid_fixture(tmp_path):
    """
    Creates initial disparity grid and save it in tmp.
    """

    def create_disparity_grid(data, disp_range, suffix_path, band=False, disp_type=rasterio.dtypes.int64):

        if not band:
            disparity_grid = xr.DataArray(data, dims=["row", "col"])
        else:
            disparity_grid = xr.DataArray(data, dims=["row", "col", "band"])

        path = tmp_path / suffix_path

        write_data_array(data_array=disparity_grid, filename=str(path), dtype=disp_type)

        return {"init": str(path), "range": disp_range}

    return create_disparity_grid


@pytest.fixture
def correct_grid(left_img_shape, create_disparity_grid_fixture):
    """Create a correct initial disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width) with alternating rows of 2, 0 and 3
    init_band = np.tile([[2], [0], [3]], (height // 3 + 1, width))[:height, :]

    return create_disparity_grid_fixture(init_band, 5, "disparity.tif")


@pytest.fixture
def second_correct_grid(left_img_shape, create_disparity_grid_fixture):
    """Create a correct initial disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width) with alternating columns of 5, -21 and -1
    init_band = np.tile([[5, -21, -1]], (height, width // 3 + 1))[:, :width]

    return create_disparity_grid_fixture(init_band, 5, "second_disparity.tif")


@pytest.fixture
def correct_grid_for_roi(left_img_shape, create_disparity_grid_fixture):
    """Create a correct initial disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width)
    init_band = np.arange(height * width).reshape((height, width))

    return create_disparity_grid_fixture(init_band, 5, "disparity.tif")


@pytest.fixture
def second_correct_grid_for_roi(left_img_shape, create_disparity_grid_fixture):
    """Create a correct initial disparity grid and save it in tmp"""

    height, width = left_img_shape

    # Array of size (height, width)
    init_band = np.arange(height * width, 0, -1).reshape((height, width))

    return create_disparity_grid_fixture(init_band, 5, "second_disparity.tif")


@pytest.fixture()
def window_size():
    return 5


@pytest.fixture()
def correct_pipeline_without_refinement(window_size):
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": window_size},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        }
    }


@pytest.fixture()
def reset_profiling():
    pandora2d.profiling.data.reset()
    pandora2d.profiling.expert_mode_config.enable = False
