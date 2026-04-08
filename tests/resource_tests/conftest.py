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
Module with global test fixtures.
"""

# pylint: disable=redefined-outer-name
import math

import numpy as np
from PIL import Image
import pytest


def pytest_addoption(parser):
    parser.addoption("--database", action="store", default=".pymon", required=False)


@pytest.fixture
def output_result_path():
    """Path to the directory containing the results"""
    return "./tests/resource_tests/result"


@pytest.fixture(params=[1, 2, 4])
def subpix(request):
    return request.param


@pytest.fixture(params=["sad", "ssd", "mutual_information", "zncc"])
def matching_cost_method(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def iterations(request):
    return request.param


@pytest.fixture
def large_image_shape():
    """Equivalent to a trishna configuration"""
    return (2000, 2000)


def reduce_image(input_path, output_path):
    """Resize the image by 2 centred"""
    data = np.asarray(Image.open(input_path))
    half_row, half_col = data.shape[0] // 2, data.shape[1] // 2
    row_margins = half_row // 2
    col_margins = half_col // 2
    image = Image.fromarray(
        data[half_row - row_margins : half_row + row_margins, half_col - col_margins : half_col + col_margins]
    )
    image.save(output_path, "png")


def tile_image(input_path, output_path, new_shape):
    """Enlarge the image until you obtain the image with the new shape"""
    image = np.asarray(Image.open(input_path))
    data = np.array(image)
    repeat_row = math.ceil(new_shape[0] / data.shape[0])
    repeat_col = math.ceil(new_shape[1] / data.shape[1])
    tiled = np.tile(data, (repeat_row, repeat_col))
    image = Image.fromarray(tiled[: new_shape[0], : new_shape[1]])
    image.save(output_path, "png")


@pytest.fixture()
def small_left_img_path(tmp_path, left_img_path):
    path = tmp_path / "left.png"
    reduce_image(left_img_path, path)
    return str(path)


@pytest.fixture()
def small_right_img_path(tmp_path, right_img_path):
    path = tmp_path / "right.png"
    reduce_image(right_img_path, path)
    return str(path)


@pytest.fixture()
def large_left_img_path(tmp_path, left_img_path, large_image_shape):
    path = tmp_path / "left.png"
    tile_image(left_img_path, path, large_image_shape)
    return str(path)


@pytest.fixture()
def large_right_img_path(tmp_path, right_img_path, large_image_shape):
    path = tmp_path / "right.png"
    tile_image(right_img_path, path, large_image_shape)
    return str(path)


@pytest.fixture
def left_img(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def right_img(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def input_cfg_for_estimation(left_img, right_img):
    return {
        "input": {
            "left": {
                "img": left_img,
                "nodata": "NaN",
            },
            "right": {
                "img": right_img,
            },
        }
    }


@pytest.fixture
def input_cfg(left_img, right_img):
    return {
        "input": {
            "left": {
                "img": left_img,
                "nodata": "NaN",
            },
            "right": {
                "img": right_img,
            },
            "col_disparity": {"init": -1, "range": 1},
            "row_disparity": {"init": -1, "range": 1},
        }
    }
