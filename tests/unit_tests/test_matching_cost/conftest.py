# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
Test Matching cost fixtures class
"""

# pylint: disable=redefined-outer-name
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr
from rasterio import Affine
from skimage.io import imsave
from pandora import import_plugin
from pandora2d import matching_cost

from pandora2d.img_tools import create_datasets_from_inputs, add_disparity_grid


@pytest.fixture()
def import_plugins():
    import_plugin()


@pytest.fixture()
def squared_image_size():
    return 10, 10


@pytest.fixture()
def left_image(tmp_path, squared_image_size):
    """
    Create a fake left image
    """
    image_path = tmp_path / "left_img.png"
    data = np.random.randint(255, size=squared_image_size, dtype=np.uint8)
    imsave(image_path, data)

    return image_path


@pytest.fixture()
def right_image(tmp_path, squared_image_size):
    """
    Create a fake right image
    """
    image_path = tmp_path / "right_img.png"
    data = np.random.randint(255, size=squared_image_size, dtype=np.uint8)
    imsave(image_path, data)

    return image_path


@pytest.fixture()
def col_disparity():
    return {"init": 1, "range": 1}


@pytest.fixture()
def row_disparity():
    return {"init": 1, "range": 2}


@pytest.fixture()
def input_config(left_image, right_image, col_disparity, row_disparity):
    return {
        "left": {
            "img": left_image,
            "nodata": -9999,
        },
        "right": {
            "img": right_image,
            "nodata": -9999,
        },
        "col_disparity": col_disparity,
        "row_disparity": row_disparity,
    }


@pytest.fixture()
def matching_cost_method():
    return "zncc"


@pytest.fixture()
def window_size():
    return 3


@pytest.fixture()
def subpix():
    return 1


@pytest.fixture()
def matching_cost_config(step, matching_cost_method, window_size, subpix):
    return {"matching_cost_method": matching_cost_method, "window_size": window_size, "step": step, "subpix": subpix}


@pytest.fixture()
def matching_cost_object(matching_cost_config):
    """
    Return the right matching cost object according to matching cost method
    """

    matching_cost_object = matching_cost.MatchingCostRegistry.get(matching_cost_config["matching_cost_method"])

    return matching_cost_object


@pytest.fixture()
def left_image_with_shift(tmp_path):
    """
    Create a fake image to test roi configuration
    """
    image_path = tmp_path / "left_img.png"
    data = np.array(
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1]]),
        dtype=np.uint8,
    )
    imsave(image_path, data)

    return image_path


@pytest.fixture()
def right_image_with_shift(tmp_path):
    """
    Create a fake image to test roi configuration
    """
    image_path = tmp_path / "right_img.png"
    data = np.array(
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.uint8,
    )
    imsave(image_path, data)

    return image_path


@pytest.fixture()
def create_image():
    """Create an image."""

    def create(data):
        return xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": -9999,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            },
        )

    return create


@pytest.fixture()
def left_zncc(create_image):
    """Left image for Zncc."""
    data = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [3, 4, 5, 6, 7],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    return create_image(data)


@pytest.fixture()
def right_zncc(create_image):
    """Right image for Zncc."""
    data = np.array(
        (
            [
                [1, 1, 1, 1, 1],
                [3, 4, 5, 6, 7],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ),
        dtype=np.float64,
    )
    return create_image(data)


@pytest.fixture()
def null_disparity_grid():
    return np.zeros((3, 3))


@pytest.fixture()
def positive_disparity_grid():
    return np.full((3, 3), 2)


@pytest.fixture()
def negative_disparity_grid():
    return np.full((3, 3), -2)


class DisparityGrids(NamedTuple):
    """
    NamedTuple used to group disparity grids together in tests.
    """

    col_min: np.ndarray
    col_max: np.ndarray
    row_min: np.ndarray
    row_max: np.ndarray


class StepData(NamedTuple):
    """NamedTuple used to group data related to Step together."""

    left: xr.Dataset
    right: xr.Dataset
    full_matching_cost: np.ndarray
    disparity_grids: DisparityGrids


@pytest.fixture()
def data_with_null_disparity(left_zncc, right_zncc, null_disparity_grid):
    """
    Coherent Data for test_step.
    """
    col_disparity_cfg = {"init": 0, "range": 0}
    row_disparity_cfg = {"init": 0, "range": 0}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [[[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]]],
            [[[np.nan]], [[-0.45]], [[-0.47058824]], [[-0.48076922]], [[np.nan]]],
            [[[np.nan]], [[-0.45]], [[-0.47058824]], [[-0.48076922]], [[np.nan]]],
            [[[np.nan]], [[0.0]], [[0.0]], [[0.0]], [[np.nan]]],
            [[[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]], [[np.nan]]],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_positive_disparity_in_col(left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid):
    """Coherent Data for test_step."""
    col_disparity_cfg = {"init": 1, "range": 1}
    row_disparity_cfg = {"init": 0, "range": 0}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=positive_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[-0.45, -0.460179, -0.46513027]],
                [[-0.47058824, -0.4756515, np.nan]],
                [[-0.48076922, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[-0.45, -0.460179, -0.46513027]],
                [[-0.47058824, -0.4756515, np.nan]],
                [[-0.48076922, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, np.nan]],
                [[0.0, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_positive_disparity_in_row(left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid):
    """Coherent Data for test_step."""
    col_disparity_cfg = {"init": 0, "range": 0}
    row_disparity_cfg = {"init": 1, "range": 1}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=positive_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[-0.45], [-0.45], [0.0]],
                [[-0.47058824], [-0.47058824], [0.0]],
                [[-0.48076922], [-0.48076922], [0.0]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[-0.45], [0.0], [np.nan]],
                [[-0.47058824], [0.0], [np.nan]],
                [[-0.48076922], [0.0], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[0.0], [np.nan], [np.nan]],
                [[0.0], [np.nan], [np.nan]],
                [[0.0], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_negative_disparity_in_col(left_zncc, right_zncc, null_disparity_grid, negative_disparity_grid):
    """Coherent Data for test_step."""
    col_disparity_cfg = {"init": -1, "range": 1}
    row_disparity_cfg = {"init": 0, "range": 0}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=negative_disparity_grid,
        col_max=null_disparity_grid,
        row_min=null_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, -0.45]],
                [[np.nan, -0.460179, -0.47058824]],
                [[-0.46513027, -0.4756515, -0.48076922]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, -0.45]],
                [[np.nan, -0.460179, -0.47058824]],
                [[-0.46513027, -0.4756515, -0.48076922]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, 0.0]],
                [[np.nan, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_negative_disparity_in_row(left_zncc, right_zncc, null_disparity_grid, negative_disparity_grid):
    """Coherent Data for test_step."""
    col_disparity_cfg = {"init": 0, "range": 0}
    row_disparity_cfg = {"init": -1, "range": 1}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=null_disparity_grid,
        row_min=negative_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [-0.45]],
                [[np.nan], [np.nan], [-0.47058824]],
                [[np.nan], [np.nan], [-0.48076922]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [1.0], [-0.45]],
                [[np.nan], [1.0], [-0.47058824]],
                [[np.nan], [1.0], [-0.48076922]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[-0.45], [1.0], [0.0]],
                [[-0.47058824], [1.0], [0.0]],
                [[-0.48076922], [1.0], [0.0]],
                [[np.nan], [np.nan], [np.nan]],
            ],
            [
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
                [[np.nan], [np.nan], [np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def data_with_disparity_negative_in_row_and_positive_in_col(
    left_zncc, right_zncc, null_disparity_grid, positive_disparity_grid, negative_disparity_grid
):
    """Coherent Data for test_step."""
    col_disparity_cfg = {"init": 1, "range": 1}
    row_disparity_cfg = {"init": -1, "range": 1}
    left_zncc.pipe(add_disparity_grid, col_disparity_cfg, row_disparity_cfg)

    disparity_grids = DisparityGrids(
        col_min=null_disparity_grid,
        col_max=positive_disparity_grid,
        row_min=negative_disparity_grid,
        row_max=null_disparity_grid,
    )
    full_matching_cost = np.array(
        [
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [-0.45, -0.460179, -0.46513027]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [-0.47058824, -0.4756515, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [-0.48076922, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [1.0, 0.99705446, 0.99227786], [-0.45, -0.460179, -0.46513027]],
                [[np.nan, np.nan, np.nan], [1.0, 0.99886817, np.nan], [-0.47058824, -0.4756515, np.nan]],
                [[np.nan, np.nan, np.nan], [1.0, np.nan, np.nan], [-0.48076922, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[-0.45, -0.460179, -0.46513027], [1.0, 0.99705446, 0.99227786], [0.0, 0.0, 0.0]],
                [[-0.47058824, -0.4756515, np.nan], [1.0, 0.99886817, np.nan], [0.0, 0.0, np.nan]],
                [[-0.48076922, np.nan, np.nan], [1.0, np.nan, np.nan], [0.0, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    return StepData(
        left=left_zncc, right=right_zncc, full_matching_cost=full_matching_cost, disparity_grids=disparity_grids
    )


@pytest.fixture()
def configuration_roi(matching_cost_config, input_config, roi):
    return {"input": input_config, "pipeline": {"matching_cost": matching_cost_config}, "ROI": roi}


@pytest.fixture()
def step():
    return [1, 1]


@pytest.fixture()
def roi(margins):
    return {"col": {"first": 2, "last": 3}, "row": {"first": 2, "last": 3}, "margins": margins}


@pytest.fixture()
def margins():
    return [1, 2, 1, 1]


@pytest.fixture()
def matching_cost_matcher(matching_cost_config):
    return matching_cost.PandoraMatchingCostMethods(matching_cost_config)


@pytest.fixture()
def cost_volumes(input_config, matching_cost_matcher, configuration):
    """Create cost_volumes."""
    img_left, img_right = create_datasets_from_inputs(input_config, roi=None)

    matching_cost_matcher.allocate(img_left=img_left, img_right=img_right, cfg=configuration)

    # compute cost volumes
    return matching_cost_matcher.compute_cost_volumes(img_left=img_left, img_right=img_right)


@pytest.fixture()
def configuration(matching_cost_config, input_config):
    return {"input": input_config, "pipeline": {"matching_cost": matching_cost_config}}
