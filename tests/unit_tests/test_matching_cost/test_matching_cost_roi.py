#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test Matching cost class
"""
# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from skimage.io import imsave

from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d import matching_cost


@pytest.fixture()
def left_image(tmp_path):
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
def right_image(tmp_path):
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


def test_roi_inside_and_margins_inside(left_image, right_image):
    """
    Test the pandora2d matching cost with roi inside the image
    """
    # input configuration
    input_cfg = {
        "input": {
            "left": {
                "img": left_image,
                "nodata": -9999,
            },
            "right": {
                "img": right_image,
                "nodata": -9999,
            },
            "col_disparity": [0, 1],
            "row_disparity": [-1, 1],
        }
    }
    input_config = input_cfg["input"]
    # read images
    img_left, img_right = create_datasets_from_inputs(input_config, roi=None)

    # Matching cost configuration
    cfg = {"matching_cost_method": "zncc", "window_size": 3}
    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg)
    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        cfg=cfg,
    )

    # compute cost volumes
    zncc = matching_cost_matcher.compute_cost_volumes(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )

    # crop image with roi
    roi = {"col": {"first": 2, "last": 3}, "row": {"first": 2, "last": 3}, "margins": [1, 2, 1, 1]}
    img_left, img_right = create_datasets_from_inputs(input_config, roi=roi)

    matching_cost_matcher.allocate_cost_volume_pandora(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        cfg=cfg,
    )
    # compute cost volumes with roi
    zncc_roi = matching_cost_matcher.compute_cost_volumes(
        img_left=img_left,
        img_right=img_right,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
    )

    assert zncc["cost_volumes"].data.shape == (5, 5, 2, 2)
    assert zncc_roi["cost_volumes"].data.shape == (5, 4, 2, 2)
    np.testing.assert_array_equal(
        zncc["cost_volumes"].data[2:4, 2:4, :, :], zncc_roi["cost_volumes"].data[2:4, 1:3, :, :]
    )
