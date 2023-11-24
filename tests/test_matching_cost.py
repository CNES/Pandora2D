#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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

# Remove this with use of fixtures
# pylint: disable=duplicate-code

import unittest

import numpy as np
import xarray as xr
import pytest
import json_checker
from rasterio import Affine
from skimage.io import imsave

from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d import matching_cost


class TestMatchingCost(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the class MatchingCost
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """
        # Create a stereo object
        data = np.array(
            ([-9999, -9999, -9999], [1, 1, 1], [3, 4, 5]),
            dtype=np.float64,
        )
        mask = np.array(([1, 1, 1], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        self.left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        self.left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            ([1, 1, 1], [3, 4, 5], [1, 1, 1]),
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        self.right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        self.right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

    @staticmethod
    def test_check_conf():
        """
        test check_conf of matching cost pipeline
        """
        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5})

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "census", "window_size": 5})

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": -1})

    @staticmethod
    def test_get_margins():
        """
        test get_margins of matching cost pipeline
        """
        gt = [2, 2, 2, 2]
        _matching_cost = matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5})

        mc_margins = _matching_cost.get_margins()

        assert len(mc_margins) == len(gt)
        assert all(a == b for a, b in zip(mc_margins, gt))

    @staticmethod
    def test_step_configuration():
        """
        Test step in matching_cost configuration
        """

        matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3]})

        # Test with a negative step : test should fail
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [-2, 3]})

        # Test with a one size list step : test should fail
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2]})

        # Test with a three elements list step : test should fail
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": [2, 3, 4]})

        # Test with a str elements list step : test should fail
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            matching_cost.MatchingCost({"matching_cost_method": "zncc", "window_size": 5, "step": ["2", 3]})

    def test_compute_cv_ssd(self):
        """
        Test the  cost volume product by ssd
        """
        # sum of squared difference images self.left, self.right, window_size=1
        cfg = {"matching_cost_method": "ssd", "window_size": 1}
        # sum of squared difference ground truth for the images self.left, self.right, window_size=1
        ad_ground_truth = np.zeros((3, 3, 2, 2))
        # disp_x = -1, disp_y = -1
        ad_ground_truth[:, :, 0, 0] = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 0, 0], [np.nan, (4 - 3) ** 2, (5 - 4) ** 2]]
        )

        # disp_x = -1, disp_y = 0
        ad_ground_truth[:, :, 0, 1] = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, (1 - 3) ** 2, (1 - 4) ** 2], [np.nan, (4 - 1) ** 2, (5 - 1) ** 2]]
        )

        # disp_x = 0, disp_y = 0
        ad_ground_truth[:, :, 1, 1] = np.array(
            [
                [np.nan, np.nan, np.nan],
                [(1 - 3) ** 2, (1 - 4) ** 2, (1 - 5) ** 2],
                [(3 - 1) ** 2, (4 - 1) ** 2, (5 - 1) ** 2],
            ]
        )

        # disp_x = 0, disp_y = -1
        ad_ground_truth[:, :, 1, 0] = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [0, 0, 0]])
        # initialise matching cost
        matching_cost_matcher = matching_cost.MatchingCost(cfg)

        # compute cost volumes
        ssd = matching_cost_matcher.compute_cost_volumes(
            img_left=self.left,
            img_right=self.right,
            grid_min_col=np.full((3, 3), -1),
            grid_max_col=np.full((3, 3), 0),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 0),
            cfg=cfg,
        )

        # check that the generated cost_volumes is equal to ground truth
        np.testing.assert_allclose(ssd["cost_volumes"].data, ad_ground_truth, atol=1e-06)

    def test_compute_cv_sad(self):
        """
        Test the  cost volume product by sad
        """

        # sum of squared difference images self.left, self.right, window_size=1
        cfg = {"matching_cost_method": "sad", "window_size": 1}
        # sum of absolute difference ground truth for the images self.left, self.right, window_size=1
        ad_ground_truth = np.zeros((3, 3, 2, 2))
        # disp_x = -1, disp_y = -1
        ad_ground_truth[:, :, 0, 0] = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 0, 0], [np.nan, np.abs(4 - 3), np.abs(5 - 4)]]
        )

        # disp_x = -1, disp_y = 0
        ad_ground_truth[:, :, 0, 1] = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, np.abs(1 - 3), np.abs(1 - 4)], [np.nan, np.abs(4 - 1), np.abs(5 - 1)]]
        )

        # disp_x = 0, disp_y = 0
        ad_ground_truth[:, :, 1, 1] = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.abs(1 - 3), np.abs(1 - 4), np.abs(1 - 5)],
                [np.abs(3 - 1), np.abs(4 - 1), np.abs(5 - 1)],
            ]
        )

        # disp_x = 0, disp_y = -1
        ad_ground_truth[:, :, 1, 0] = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [0, 0, 0]])

        # initialise matching cost
        matching_cost_matcher = matching_cost.MatchingCost(cfg)
        # compute cost volumes
        sad = matching_cost_matcher.compute_cost_volumes(
            img_left=self.left,
            img_right=self.right,
            grid_min_col=np.full((3, 3), -1),
            grid_max_col=np.full((3, 3), 0),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 0),
            cfg=cfg,
        )
        # check that the generated cost_volumes is equal to ground truth
        np.testing.assert_allclose(sad["cost_volumes"].data, ad_ground_truth, atol=1e-06)

    @staticmethod
    def test_compute_cv_zncc():
        """
        Test the  cost volume product by zncc
        """
        data = np.array(
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            dtype=np.float64,
        )
        mask = np.array(
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
        )
        left_zncc = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left_zncc.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            ([[1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            dtype=np.float64,
        )
        mask = np.array(
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
        )
        right_zncc = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right_zncc.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        # sum of squared difference images self.left, self.right, window_size=3
        cfg = {"matching_cost_method": "zncc", "window_size": 3}
        # sum of absolute difference ground truth for the images self.left, self.right, window_size=1

        left = left_zncc["im"].data
        right = right_zncc["im"].data
        right_shift = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [1, 1, 1, 1, 1],
                [3, 4, 5, 6, 7],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        # row = 1, col = 1, disp_x = 0, disp_y = 0, ground truth equal -0,45
        ad_ground_truth_1_1_0_0 = (
            np.mean(left[0:3, 0:3] * right[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right[0:3, 0:3]))
        ) / (np.std(left[0:3, 0:3]) * np.std(right[0:3, 0:3]))
        # row = 1, col = 1, disp_x = 0, disp_y = -1, , ground truth equal NaN
        ad_ground_truth_1_1_0_1 = (
            np.mean(left[0:3, 0:3] * right_shift[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right_shift[0:3, 0:3]))
        ) / (np.std(left[0:3, 0:3]) * np.std(right_shift[0:3, 0:3]))
        # row = 2, col = 2, disp_x = 0, disp_y = 0, , ground truth equal -0,47
        ad_ground_truth_2_2_0_0 = (
            np.mean(left[1:4, 1:4] * right[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right[1:4, 1:4]))
        ) / (np.std(left[1:4, 1:4]) * np.std(right[1:4, 1:4]))
        # row = 2, col = 2, disp_x = 0, disp_y = -1, ground truth equal 1
        ad_ground_truth_2_2_0_1 = (
            np.mean(left[1:4, 1:4] * right_shift[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right_shift[1:4, 1:4]))
        ) / (np.std(left[1:4, 1:4]) * np.std(right_shift[1:4, 1:4]))

        # initialise matching cost
        matching_cost_matcher = matching_cost.MatchingCost(cfg)
        # compute cost volumes
        zncc = matching_cost_matcher.compute_cost_volumes(
            img_left=left_zncc,
            img_right=right_zncc,
            grid_min_col=np.full((3, 3), 0),
            grid_max_col=np.full((3, 3), 1),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 0),
            cfg=cfg,
        )
        # check that the generated cost_volumes is equal to ground truth

        np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 1], ad_ground_truth_1_1_0_0, rtol=1e-06)
        np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 0], ad_ground_truth_1_1_0_1, rtol=1e-06)
        np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 1], ad_ground_truth_2_2_0_0, rtol=1e-06)
        np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 0], ad_ground_truth_2_2_0_1, rtol=1e-06)

    def test_allocate_cost_volume(self):
        """
        Test the allocate cost_volumes function
        """

        # generated data for the test
        np_data = np.empty((3, 3, 3, 3))
        np_data.fill(np.nan)

        c_row = [0, 1, 2]
        c_col = [0, 1, 2]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)
        col = np.arange(c_col[0], c_col[-1] + 1)

        disparity_range_col = np.arange(-1, 1 + 1)
        disparity_range_row = np.arange(-1, 1 + 1)

        # Create the cost volume
        if np_data is None:
            np_data = np.zeros(
                (len(row), len(col), len(disparity_range_col), len(disparity_range_row)), dtype=np.float32
            )

        cost_volumes_test = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], np_data)},
            coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
        )

        cost_volumes_test.attrs["measure"] = "zncc"
        cost_volumes_test.attrs["window_size"] = 3
        cost_volumes_test.attrs["type_measure"] = "max"
        cost_volumes_test.attrs["subpixel"] = 1
        cost_volumes_test.attrs["offset_row_col"] = 1
        cost_volumes_test.attrs["cmax"] = 1
        cost_volumes_test.attrs["crs"] = None
        cost_volumes_test.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        cost_volumes_test.attrs["band_correl"] = None

        # data by function compute_cost_volume
        cfg = {"matching_cost_method": "zncc", "window_size": 3}
        matching_cost_matcher = matching_cost.MatchingCost(cfg)

        cost_volumes_fun = matching_cost_matcher.compute_cost_volumes(
            img_left=self.left,
            img_right=self.right,
            grid_min_col=np.full((3, 3), -1),
            grid_max_col=np.full((3, 3), 1),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 1),
            cfg=cfg,
        )

        # check that the generated xarray dataset is equal to the ground truth
        np.testing.assert_array_equal(cost_volumes_fun["cost_volumes"].data, cost_volumes_test["cost_volumes"].data)
        assert cost_volumes_fun.attrs == cost_volumes_test.attrs

def test_step():
    """We expect step to work."""

    data = np.array(
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    left_zncc = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left_zncc.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    data = np.array(
        ([[1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    right_zncc = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right_zncc.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    # sum of squared difference images self.left, self.right, window_size=3
    cfg = {"matching_cost_method": "zncc", "window_size": 3, "step":[1,2]}
    # sum of absolute difference ground truth for the images self.left, self.right, window_size=1

    left = left_zncc["im"].data
    right = right_zncc["im"].data
    right_shift = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, 1, 1, 1, 1],
            [3, 4, 5, 6, 7],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    # row = 1, col = 1, disp_x = 0, disp_y = 0, ground truth equal -0,45
    ad_ground_truth_1_1_0_0 = (
                                      np.mean(left[0:3, 0:3] * right[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right[0:3, 0:3]))
                              ) / (np.std(left[0:3, 0:3]) * np.std(right[0:3, 0:3]))
    # row = 1, col = 1, disp_x = 0, disp_y = -1, , ground truth equal NaN
    ad_ground_truth_1_1_0_1 = (
                                      np.mean(left[0:3, 0:3] * right_shift[0:3, 0:3]) - (np.mean(left[0:3, 0:3]) * np.mean(right_shift[0:3, 0:3]))
                              ) / (np.std(left[0:3, 0:3]) * np.std(right_shift[0:3, 0:3]))
    # row = 2, col = 2, disp_x = 0, disp_y = 0, , ground truth equal -0,47
    ad_ground_truth_2_2_0_0 = (
                                      np.mean(left[1:4, 1:4] * right[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right[1:4, 1:4]))
                              ) / (np.std(left[1:4, 1:4]) * np.std(right[1:4, 1:4]))
    # row = 2, col = 2, disp_x = 0, disp_y = -1, ground truth equal 1
    ad_ground_truth_2_2_0_1 = (
                                      np.mean(left[1:4, 1:4] * right_shift[1:4, 1:4]) - (np.mean(left[1:4, 1:4]) * np.mean(right_shift[1:4, 1:4]))
                              ) / (np.std(left[1:4, 1:4]) * np.std(right_shift[1:4, 1:4]))

    # initialise matching cost
    matching_cost_matcher = matching_cost.MatchingCost(cfg)
    # compute cost volumes
    zncc = matching_cost_matcher.compute_cost_volumes(
        img_left=left_zncc,
        img_right=right_zncc,
        grid_min_col=np.full((3, 3), 0),
        grid_max_col=np.full((3, 3), 1),
        grid_min_row=np.full((3, 3), -1),
        grid_max_row=np.full((3, 3), 0),
        cfg=cfg,
    )
    # check that the generated cost_volumes is equal to ground truth

    np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 1], ad_ground_truth_1_1_0_0, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[1, 1, 0, 0], ad_ground_truth_1_1_0_1, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 1], ad_ground_truth_2_2_0_0, rtol=1e-06)
    np.testing.assert_allclose(zncc["cost_volumes"].data[2, 2, 0, 0], ad_ground_truth_2_2_0_1, rtol=1e-06)


class TestMatchingCostWithRoi:
    """Test using roi in pandora2d processing"""

    @pytest.fixture()
    def left_image(self, tmp_path):
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
    def right_image(self, tmp_path):
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

    @staticmethod
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

        # compute cost volumes
        zncc = matching_cost_matcher.compute_cost_volumes(
            img_left=img_left,
            img_right=img_right,
            grid_min_col=np.full((3, 3), 0),
            grid_max_col=np.full((3, 3), 1),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 0),
            cfg=cfg,
        )

        # crop image with roi
        roi = {"col": {"first": 2, "last": 3}, "row": {"first": 2, "last": 3}, "margins": [1, 2, 1, 1]}
        img_left, img_right = create_datasets_from_inputs(input_config, roi=roi)

        # compute cost volumes with roi
        zncc_roi = matching_cost_matcher.compute_cost_volumes(
            img_left=img_left,
            img_right=img_right,
            grid_min_col=np.full((3, 3), 0),
            grid_max_col=np.full((3, 3), 1),
            grid_min_row=np.full((3, 3), -1),
            grid_max_row=np.full((3, 3), 0),
            cfg=cfg,
        )

        assert zncc["cost_volumes"].data.shape == (5, 5, 2, 2)
        assert zncc_roi["cost_volumes"].data.shape == (5, 4, 2, 2)
        np.testing.assert_array_equal(
            zncc["cost_volumes"].data[2:4, 2:4, :, :], zncc_roi["cost_volumes"].data[2:4, 1:3, :, :]
        )
