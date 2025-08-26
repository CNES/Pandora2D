#!/usr/bin/env python
#
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
Tests pandora2d machine execution with mutual information and zncc cpp methods
"""

# pylint: disable=redefined-outer-name
import time
import pytest
import numpy as np

import pandora2d
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing


@pytest.fixture()
def float_precision():
    return "float32"


@pytest.fixture()
def cfg_for_correlation(correct_input_for_functional_tests, method, window_size, subpix, step, float_precision):
    """
    Return user configuration to test mutual information and zncc methods
    """

    user_cfg = {
        **correct_input_for_functional_tests,
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": method,
                "window_size": window_size,
                "subpix": subpix,
                "step": step,
                "float_precision": float_precision,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
        },
        "output": {"path": "where"},
    }

    return user_cfg


@pytest.fixture()
def cfg_for_correlation_with_roi(cfg_for_correlation, roi):
    """
    Return user configuration to test mutual information and zncc cpp methods with ROI
    """

    cfg_for_correlation["ROI"] = roi

    return cfg_for_correlation


class TestCorrelation:
    """
    Test that the pandora2d machine runs correctly with the mutual information and zncc cpp methods
    for different parameter panels
    """

    @pytest.mark.parametrize("method", ["mutual_information", "zncc"])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize("window_size", [1, 3, 5])
    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
    @pytest.mark.parametrize("roi", [{"col": {"first": 100, "last": 120}, "row": {"first": 100, "last": 120}}])
    @pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 1}])
    @pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 3}])
    @pytest.mark.parametrize("float_precision", ["float64", "float32"])
    def test_correlation_execution(self, float_precision, cfg_for_correlation_with_roi):
        """
        Description : Test that execution of Pandora2d with mutual information and zncc cpp does not fail.
        Data :
            * Left_img : cones/monoband/left.png
            * Right_img : cones/monoband/right.png
        """
        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(cfg_for_correlation_with_roi, pandora2d_machine)

        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

        dataset_disp_maps, _ = pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

        # Checking that resulting disparity maps are not full of nans
        assert not np.all(np.isnan(dataset_disp_maps.row_map.data))
        assert not np.all(np.isnan(dataset_disp_maps.col_map.data))
        assert pandora2d_machine.cost_volumes["cost_volumes"].data.dtype == np.dtype(float_precision)

    @pytest.mark.parametrize("method", ["mutual_information", "zncc"])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize("window_size", [1, 3, 5])
    @pytest.mark.parametrize("step", [[1, 1], [2, 1], [1, 3], [5, 5]])
    @pytest.mark.parametrize("roi", [{"col": {"first": 100, "last": 120}, "row": {"first": 100, "last": 120}}])
    @pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 1}])
    @pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 3}])
    def test_invalid_points_not_computed(self, cfg_for_correlation_with_roi):
        """
        Description : Test that when running the matching cost step with mutual information and zncc cpp,
        invalid points are not computed.
        Data :
            * Left_img : cones/monoband/left.png
            * Right_img : cones/monoband/right.png
        """

        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(cfg_for_correlation_with_roi, pandora2d_machine)

        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

        # Run matching cost step
        pandora2d_machine.run_prepare(image_datasets.left, image_datasets.right, cfg)
        pandora2d_machine.run("matching_cost", cfg)

        invalid_point = np.where(pandora2d_machine.cost_volumes["criteria"].data != 0)
        assert np.all(pandora2d_machine.cost_volumes["cost_volumes"].data[invalid_point] == 0)

    @pytest.mark.parametrize("method", ["mutual_information"])
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("window_size", [5])
    @pytest.mark.parametrize("step", [[1, 1]])
    @pytest.mark.parametrize("roi", [{"col": {"first": 100, "last": 150}, "row": {"first": 100, "last": 150}}])
    @pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 1}])
    @pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 3}])
    def test_computation_time_with_mask(self, cfg_for_correlation_with_roi, full_invalid_mask_path):
        """
        Description : Test that the matching cost step with mutual information is faster when
        using an input mask. (Some points are not computed in this case)
        Data :
            * Left_img : cones/monoband/left.png
            * Right_img : cones/monoband/right.png
        """

        # Computation without mask

        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(cfg_for_correlation_with_roi, pandora2d_machine)

        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

        # Run matching cost step
        pandora2d_machine.run_prepare(image_datasets.left, image_datasets.right, cfg)
        start_time = time.time()
        pandora2d_machine.run("matching_cost", cfg)
        duration = time.time() - start_time

        # Computation with mask

        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(cfg_for_correlation_with_roi, pandora2d_machine)

        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

        cfg["input"]["left"]["mask"] = full_invalid_mask_path

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"], roi=roi)

        # Run matching cost step
        pandora2d_machine.run_prepare(image_datasets.left, image_datasets.right, cfg)
        start_time_mask = time.time()
        pandora2d_machine.run("matching_cost", cfg)
        duration_mask = time.time() - start_time_mask

        # Check that the more invalid points, the faster the mutual information computation.
        assert duration > duration_mask


class TestNbBinsMax:
    """
    Test that the pandora2d machine runs correctly with the mutual information method
    and images for which nb_bins reaches NB_BINS_MAX
    """

    @pytest.fixture()
    def method(self):
        return "mutual_information"

    @pytest.fixture(scope="session")
    def left_img_path(self, root_dir):
        return str(root_dir / "tests/functional_tests/matching_cost/data/img_nb_bins_max.tif")

    @pytest.fixture(scope="session")
    def right_img_path(self, root_dir):
        return str(root_dir / "tests/functional_tests/matching_cost/data/img_nb_bins_max.tif")

    @pytest.mark.parametrize("subpix", [4])
    @pytest.mark.parametrize("window_size", [65])
    @pytest.mark.parametrize("step", [[32, 32]])
    @pytest.mark.parametrize("col_disparity", [{"init": 0, "range": 10}])
    @pytest.mark.parametrize("row_disparity", [{"init": 0, "range": 10}])
    def test_nb_bins_max(self, cfg_for_correlation):
        """
        Description : Test that execution of Pandora2d with mutual information
        and image with nb_bins=NB_BINS_MAX does not fail.
        Data :
            * Left_img : tests/functional_tests/matching_cost/data/img_nb_bins_max.tif
            * Right_img : tests/functional_tests/matching_cost/data/img_nb_bins_max.tif
        """
        pandora2d_machine = Pandora2DMachine()

        cfg = check_conf(cfg_for_correlation, pandora2d_machine)

        image_datasets = create_datasets_from_inputs(input_config=cfg["input"])

        dataset_disp_maps, _ = pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

        # Checking that resulting disparity maps are not full of nans
        assert not np.all(np.isnan(dataset_disp_maps.row_map.data))
        assert not np.all(np.isnan(dataset_disp_maps.col_map.data))
