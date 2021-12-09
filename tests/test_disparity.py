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
Test Disparity class
"""

import unittest
import pytest
import numpy as np
import xarray as xr
from rasterio import Affine
import json_checker

from pandora2d import matching_cost, disparity


class TestDisparity(unittest.TestCase):
    """
    TestDisparity class allows to test all the methods in the class Disparity
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
        data = np.array(
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            dtype=np.float64,
        )
        mask = np.array(
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
        )
        self.left_arg = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        self.left_arg.attrs = {
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
        self.right_arg = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        self.right_arg.attrs = {
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
        disparity.Disparity(**{"disparity_method": "wta", "invalid_disparity": -9999}) # type: ignore

        with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError):
            disparity.Disparity(**{"invalid_disparity": "5"}) # type: ignore

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            disparity.Disparity(**{"disparity_method": "WTN"}) # type: ignore

    def test_min_split(self):
        """
        Test the min_split function
        """

        # create a cost_volume, with SAD measure, window_size 1, dispx_min 0, dispx_max 1, dispy_min 0, dispy_max 1
        cfg = {"matching_cost_method": "sad", "window_size": 1}
        matching_cost_test = matching_cost.MatchingCost(**cfg) # type: ignore

        cvs = matching_cost_test.compute_cost_volumes(self.left, self.right, 0, 1, -1, 0, **cfg) # type: ignore

        ad_ground_truth = np.zeros((3, 3, 2))
        ad_ground_truth[:, :, 0] = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [0, 0, 0]])
        ad_ground_truth[:, :, 1] = np.array([[np.nan, np.nan, np.nan], [0, 0, np.nan], [1, 1, np.nan]])

        disparity_test = disparity.Disparity(**{"disparity_method": "wta", "invalid_disparity": -9999}) # type: ignore
        # searching along dispy axis
        cvs_min = disparity_test.min_split(cvs, 3)

        np.testing.assert_allclose(cvs_min[:, :, 0], ad_ground_truth[:, :, 0], atol=1e-06)
        np.testing.assert_allclose(cvs_min[:, :, 1], ad_ground_truth[:, :, 1], atol=1e-06)

    def test_max_split(self):
        """
        Test the min_split function
        """

        # create a cost_volume, with SAD measure, window_size 1, dispx_min 0, dispx_max 1, dispy_min 0, dispy_max 1
        cfg = {"matching_cost_method": "sad", "window_size": 1}
        matching_cost_test = matching_cost.MatchingCost(**cfg) # type: ignore

        cvs = matching_cost_test.compute_cost_volumes(self.left, self.right, 0, 1, -1, 0, **cfg) # type: ignore

        ad_ground_truth = np.zeros((3, 3, 2))
        ad_ground_truth[:, :, 0] = np.array([[np.nan, np.nan, np.nan], [2, 3, 4], [2, 3, 4]])
        ad_ground_truth[:, :, 1] = np.array([[np.nan, np.nan, np.nan], [3, 4, np.nan], [2, 3, np.nan]])

        disparity_test = disparity.Disparity(**{"disparity_method": "wta", "invalid_disparity": -9999}) # type: ignore
        # searching along dispy axis
        cvs_max = disparity_test.max_split(cvs, 3)

        np.testing.assert_allclose(cvs_max[:, :, 0], ad_ground_truth[:, :, 0], atol=1e-06)
        np.testing.assert_allclose(cvs_max[:, :, 1], ad_ground_truth[:, :, 1], atol=1e-06)

    def test_argmin_split(self):
        """
        Test the argmin_split function
        """

        # create a cost_volume, with SAD measure, window_size 1, dispx_min 0, dispx_max 1, dispy_min 0, dispy_max 1
        cfg = {"matching_cost_method": "sad", "window_size": 3}
        matching_cost_test = matching_cost.MatchingCost(**cfg) # type: ignore

        cvs = matching_cost_test.compute_cost_volumes(self.left_arg, self.right_arg, 0, 1, -1, 0, **cfg) # type: ignore

        ad_ground_truth = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
        )

        disparity_test = disparity.Disparity(**{"disparity_method": "wta", "invalid_disparity": -9999}) # type: ignore
        # searching along dispy axis
        cvs_max = disparity_test.min_split(cvs, 3)
        min_tensor = disparity_test.argmin_split(cvs_max, 2)

        np.testing.assert_allclose(min_tensor, ad_ground_truth, atol=1e-06)

    def test_argmax_split(self):
        """
        Test the argmax_split function
        """

        # create a cost_volume, with SAD measure, window_size 1, dispx_min 0, dispx_max 1, dispy_min 0, dispy_max 1
        cfg = {"matching_cost_method": "sad", "window_size": 3}
        matching_cost_test = matching_cost.MatchingCost(**cfg) # type: ignore

        cvs = matching_cost_test.compute_cost_volumes(self.left_arg, self.right_arg, 0, 1, -1, 0, **cfg) # type: ignore

        ad_ground_truth = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
        )

        disparity_test = disparity.Disparity(**{"disparity_method": "wta", "invalid_disparity": -9999}) # type: ignore
        # searching along dispy axis
        cvs_max = disparity_test.max_split(cvs, 3)
        max_tensor = disparity_test.argmax_split(cvs_max, 2)

        np.testing.assert_allclose(max_tensor, ad_ground_truth, atol=1e-06)

    @staticmethod
    def test_compute_disparity_map_row():
        """
        Test function for disparity computation
        """
        data = np.array(
            ([[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]]),
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            [[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]],
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        ground_truth_col = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        ground_truth_row = np.array([[2, 2, 2, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]])

        # create matching_cost object with measure = ssd, window_size = 3
        cfg_mc = {"matching_cost_method": "ssd", "window_size": 1}
        matching_cost_matcher = matching_cost.MatchingCost(**cfg_mc) # type: ignore
        # create disparity object with WTA method
        cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
        disparity_matcher = disparity.Disparity(**cfg_disp) # type: ignore

        cvs = matching_cost_matcher.compute_cost_volumes(
            left, right, min_col=-2, max_col=2, min_row=-2, max_row=2, **cfg_mc
        ) # type: ignore

        delta_x, delta_y = disparity_matcher.compute_disp_maps(cvs)

        np.testing.assert_array_equal(ground_truth_col, delta_x)
        np.testing.assert_array_equal(ground_truth_row, delta_y)

    @staticmethod
    def test_compute_disparity_map_col():
        """
        Test function for disparity computation
        """
        data = np.array(
            ([[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12]]),
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            [[8, 5, 6, 7], [4, 1, 2, 3], [12, 9, 10, 11]],
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        ground_truth_row = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        ground_truth_col = np.array([[1, 1, 1, -3], [1, 1, 1, -3], [1, 1, 1, -3]])

        # create matching_cost object with measure = ssd, window_size = 3
        cfg_mc = {"matching_cost_method": "ssd", "window_size": 1}
        matching_cost_matcher = matching_cost.MatchingCost(**cfg_mc) # type: ignore
        # create disparity object with WTA method
        cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
        disparity_matcher = disparity.Disparity(**cfg_disp) # type: ignore

        cvs = matching_cost_matcher.compute_cost_volumes(
            left, right, min_col=-3, max_col=3, min_row=-3, max_row=3, **cfg_mc
        ) # type: ignore

        delta_x, delta_y = disparity_matcher.compute_disp_maps(cvs)

        np.testing.assert_array_equal(ground_truth_col, delta_x)
        np.testing.assert_array_equal(ground_truth_row, delta_y)

    @staticmethod
    def test_compute_disparity_map_col_row():
        """
        Test function for disparity computation
        """
        data = np.array(
            ([[9, 10, 11, 12], [5, 6, 7, 8], [1, 2, 3, 4]]),
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            [[8, 5, 6, 7], [4, 1, 2, 3], [12, 9, 10, 11]],
            dtype=np.float64,
        )
        mask = np.array(([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        ground_truth_row = np.array([[2, 2, 2, 2], [-1, -1, -1, -1], [-1, -1, -1, -1]])

        ground_truth_col = np.array([[1, 1, 1, -3], [1, 1, 1, -3], [1, 1, 1, -3]])

        # create matching_cost object with measure = ssd, window_size = 3
        cfg_mc = {"matching_cost_method": "ssd", "window_size": 1}
        matching_cost_matcher = matching_cost.MatchingCost(**cfg_mc) # type: ignore
        # create disparity object with WTA method
        cfg_disp = {"disparity_method": "wta", "invalid_disparity": -5}
        disparity_matcher = disparity.Disparity(**cfg_disp) # type: ignore

        cvs = matching_cost_matcher.compute_cost_volumes(
            left, right, min_col=-3, max_col=3, min_row=-3, max_row=3, **cfg_mc
        )

        delta_x, delta_y = disparity_matcher.compute_disp_maps(cvs)

        np.testing.assert_array_equal(ground_truth_col, delta_x)
        np.testing.assert_array_equal(ground_truth_row, delta_y)

    @staticmethod
    def test_masked_nan():
        """
        Test the capacity of disparity_computation to find nans
        """
        cv = np.zeros((4, 5, 2, 2))
        # disp_x = -1, disp_y = -1
        cv[:, :, 0, 0] = np.array(
            [[np.nan, np.nan, np.nan, 6, 8], [np.nan, 0, 0, np.nan, 5], [1, 1, 1, 1, 1], [1, np.nan, 2, 3, np.nan]]
        )

        # disp_x = -1, disp_y = 0
        cv[:, :, 0, 1] = np.array(
            [[np.nan, np.nan, np.nan, 1, 2], [np.nan, 2, 2, 3, 6], [4, np.nan, 1, 1, 1], [6, 6, 6, 6, np.nan]]
        )

        # disp_x = 0, disp_y = 0
        cv[:, :, 1, 1] = np.array(
            [[np.nan, np.nan, np.nan, 0, 4], [np.nan, np.nan, 3, 3, 3], [2, np.nan, 4, 4, 5], [1, 2, 3, 4, np.nan]]
        )

        # disp_x = 0, disp_y = -1
        cv[:, :, 1, 0] = np.array(
            [[np.nan, np.nan, np.nan, 5, 60], [np.nan, 7, 8, 9, 10], [np.nan, np.nan, 6, 10, 11], [7, 8, 9, 10, np.nan]]
        )

        c_row = [0, 1, 2, 3]
        c_col = [0, 1, 2, 3, 4]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)
        col = np.arange(c_col[0], c_col[-1] + 1)

        disparity_range_col = np.arange(-1, 0 + 1)
        disparity_range_row = np.arange(-1, 0 + 1)

        cost_volumes_dataset = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
            coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
        )

        cost_volumes_dataset.attrs["type_measure"] = "max"

        cfg_disp = {"disparity_method": "wta", "invalid_disparity": -99}
        disparity_matcher = disparity.Disparity(**cfg_disp) # type: ignore

        ground_truth_col = np.array([[-99, -99, -99, -1, 0], [-99, 0, 0, 0, 0], [-1, -1, 0, 0, 0], [0, 0, 0, 0, -99]])

        ground_truth_row = np.array(
            [[-99, -99, -99, -1, -1], [-99, -1, -1, -1, -1], [0, -1, -1, -1, -1], [-1, -1, -1, -1, -99]]
        )

        delta_x, delta_y = disparity_matcher.compute_disp_maps(cost_volumes_dataset)

        np.testing.assert_array_equal(ground_truth_col, delta_x)
        np.testing.assert_array_equal(ground_truth_row, delta_y)
