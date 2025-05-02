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
"""
Test the refinement.dichotomy pipeline.
"""

import copy
import pytest

import numpy as np


import pandora2d
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name


@pytest.fixture()
def make_cfg_for_dichotomy(
    correct_input_for_functional_tests,
    dicho_method,
    filter_method,
    subpix,
    step,
    iterations,
    roi,
):
    """
    Creates user configuration to test dichotomy loop
    """

    user_cfg = {
        **correct_input_for_functional_tests,
        "ROI": roi,
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": 7,
                "subpix": subpix,
                "step": step,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
            "refinement": {
                "refinement_method": dicho_method,
                "iterations": iterations,
                "filter": {"method": filter_method},
            },
        },
        "output": {"path": "home"},
    }

    return user_cfg


@pytest.mark.parametrize(
    ("dicho_method", "filter_method"),
    [
        ("dichotomy_python", "bicubic_python"),
        ("dichotomy_python", "sinc_python"),
        ("dichotomy", "sinc"),
        ("dichotomy", "bicubic"),
    ],
)
@pytest.mark.parametrize("subpix", [1, 2, 4])
@pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("roi", [{"col": {"first": 100, "last": 120}, "row": {"first": 100, "last": 120}}])
@pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 1}])
@pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 3}])
def test_dichotomy_execution(make_cfg_for_dichotomy):
    """
    Description : Test that execution of Pandora2d with a dichotomy refinement does not fail.
    Data :
           * Left_img : cones/monoband/left.png
           * Right_img : cones/monoband/right.png
    Requirement :
           * EX_REF_BCO_00
           * EX_REF_SINC_00
    """
    pandora2d_machine = Pandora2DMachine()

    cfg = check_conf(make_cfg_for_dichotomy, pandora2d_machine)

    cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
    roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

    image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

    dataset_disp_maps, _ = pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

    # Checking that resulting disparity maps are not full of nans
    with np.testing.assert_raises(AssertionError):
        assert np.all(np.isnan(dataset_disp_maps.row_map.data))
        assert np.all(np.isnan(dataset_disp_maps.col_map.data))


@pytest.mark.parametrize(
    ("dicho_method", "filter_method"),
    [
        ("dichotomy_python", "bicubic_python"),
        ("dichotomy_python", "sinc_python"),
        ("dichotomy", "bicubic"),
        ("dichotomy", "sinc"),
    ],
)
@pytest.mark.parametrize("subpix", [1])
@pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
@pytest.mark.parametrize("iterations", [1, 2])
# This ROI has been chosen because its corresponding disparity maps
# contain extrema disparity range values and subpixel values after refinement.
@pytest.mark.parametrize("roi", [{"col": {"first": 30, "last": 40}, "row": {"first": 160, "last": 170}}])
# We use small disparity intervals to obtain extrema of disparity ranges in the disparity maps.
# Once the variable disparity grids have been introduced into pandora2d,
# this type of disparity will also need to be tested here.
@pytest.mark.parametrize("col_disparity", [{"init": -1, "range": 1}])
@pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 1}])
def test_extrema_disparities_not_processed(make_cfg_for_dichotomy):
    """
    Description : Test that execution of Pandora2d with a dichotomy refinement does not
    take into account points for which best cost value is found at the edge of the disparity range.
    Data :
           * Left_img : cones/monoband/left.png
           * Right_img : cones/monoband/right.png
    """
    pandora2d_machine = pandora2d.state_machine.Pandora2DMachine()

    cfg = check_conf(make_cfg_for_dichotomy, pandora2d_machine)

    cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
    roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

    image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

    # Prepare Pandora2D machine
    pandora2d_machine.run_prepare(image_datasets.left, image_datasets.right, cfg)
    # Run matching cost step
    pandora2d_machine.run("matching_cost", cfg)
    # Run disparity step
    pandora2d_machine.run("disparity", cfg)
    # Make a copy of disparity maps before refinement step
    copy_disp_maps = copy.deepcopy(pandora2d_machine.dataset_disp_maps)
    # Run refinement step
    pandora2d_machine.run("refinement", cfg)

    # Select correct rows and columns in case of a step different from 1.
    row_cv = pandora2d_machine.cost_volumes.row.values
    col_cv = pandora2d_machine.cost_volumes.col.values

    # Get points for which best cost value is at the edge of the row disparity range
    mask_min_row = np.nonzero(
        copy_disp_maps["row_map"].data == image_datasets.left.sel(row=row_cv, col=col_cv).row_disparity[0, :, :]
    )
    mask_max_row = np.nonzero(
        copy_disp_maps["row_map"].data == image_datasets.left.sel(row=row_cv, col=col_cv).row_disparity[1, :, :]
    )

    # Get points for which best cost value is at the edge of the column disparity range
    mask_min_col = np.nonzero(
        copy_disp_maps["col_map"].data == image_datasets.left.sel(row=row_cv, col=col_cv).col_disparity[0, :, :]
    )
    mask_max_col = np.nonzero(
        copy_disp_maps["col_map"].data == image_datasets.left.sel(row=row_cv, col=col_cv).col_disparity[1, :, :]
    )

    # Checking that best row disparity is unchanged for points having best cost value at the edge of row disparity range
    assert np.all(
        pandora2d_machine.dataset_disp_maps["row_map"].data[mask_min_row[0], mask_min_row[1]]
        == image_datasets.left.row_disparity.data[0, mask_min_row[0], mask_min_row[1]]
    )
    assert np.all(
        pandora2d_machine.dataset_disp_maps["row_map"].data[mask_max_row[0], mask_max_row[1]]
        == image_datasets.left.row_disparity.data[1, mask_max_row[0], mask_max_row[1]]
    )

    # Checking that best col disparity is unchanged for points having best cost value at the edge of col disparity range
    assert np.all(
        pandora2d_machine.dataset_disp_maps["col_map"].data[mask_min_col[0], mask_min_col[1]]
        == image_datasets.left.col_disparity.data[0, mask_min_col[0], mask_min_col[1]]
    )
    assert np.all(
        pandora2d_machine.dataset_disp_maps["col_map"].data[mask_max_col[0], mask_max_col[1]]
        == image_datasets.left.col_disparity.data[1, mask_max_col[0], mask_max_col[1]]
    )
