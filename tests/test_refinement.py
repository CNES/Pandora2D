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
Test refinement step
"""

import unittest
import numpy as np
import xarray as xr
import pytest
from rasterio import Affine

from pandora2d import refinement, common


class TestRefinement(unittest.TestCase):
    """
    TestRefinement class allows to test the refinement module
    """

    def SetUp(self) -> None:
        """
        Method called to prepare the test fixture
        """
        data_left = np.array([[1056, 1064, 1073],
                              [1064, 1060, 1074],
                              [1060, 1063, 1084]])

        data_right = np.array([[1061, 1082, 1115],
                               [1064, 1094, 1137],
                               [1065, 1095, 1131]])

        mask = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]),
                        dtype=np.int16)

        self.left = xr.Dataset(
            {"im": (["row", "col"], data_left), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data_left.shape[0]), "col": np.arange(data_left.shape[1])},
        )
        self.left.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        self.right = xr.Dataset(
            {"im": (["row", "col"], data_right), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data_right.shape[0]), "col": np.arange(data_right.shape[1])},
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
        Test the refinements methods
        """

        refinement.AbstractRefinement(**{"refinement_method": "interpolation"})   # type: ignore
        refinement.AbstractRefinement(**{"refinement_method": "optical_flow"})  # type: ignore

        with pytest.raises(KeyError):
            refinement.AbstractRefinement(**{"refinement_method": "wta"})   # type: ignore

    @staticmethod
    def test_refinement_method_subpixel():
        """
        test refinement
        """

        cv = np.zeros((3, 3, 5, 5))
        cv[:, :, 2, 2] = np.ones([3, 3])
        cv[:, :, 2, 3] = np.ones([3, 3])
        cv[:, :, 3, 2] = np.ones([3, 3])
        cv[:, :, 3, 3] = np.ones([3, 3])

        c_row = [0, 1, 2]
        c_col = [0, 1, 2]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)
        col = np.arange(c_col[0], c_col[-1] + 1)

        disparity_range_col = np.arange(-2, 2 + 1)
        disparity_range_row = np.arange(-2, 2 + 1)

        cost_volumes_test = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
            coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
        )

        cost_volumes_test.attrs["measure"] = "zncc"
        cost_volumes_test.attrs["window_size"] = 1
        cost_volumes_test.attrs["type_measure"] = "max"

        data = np.array(
            ([[0.4833878, 0.4833878, 0.4833878], [0.4833878, 0.4833878, 0.4833878], [0.4833878, 0.4833878, 0.4833878]]),
            dtype=np.float64,
        )

        dataset_disp_map = common.dataset_disp_maps(data, data)

        test = refinement.AbstractRefinement(**{"refinement_method": "interpolation"}) # type: ignore
        delta_x, delta_y = test.refinement_method(cost_volumes_test, dataset_disp_map)

        np.testing.assert_allclose(data, delta_y, rtol=1e-06)
        np.testing.assert_allclose(data, delta_x, rtol=1e-06)

    @staticmethod
    def test_refinement_method_pixel():
        """
        test refinement
        """

        cv = np.zeros((3, 3, 5, 5))
        cv[:, :, 1, 3] = np.ones([3, 3])

        c_row = [0, 1, 2]
        c_col = [0, 1, 2]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)
        col = np.arange(c_col[0], c_col[-1] + 1)

        disparity_range_col = np.arange(-2, 2 + 1)
        disparity_range_row = np.arange(-2, 2 + 1)

        cost_volumes_test = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
            coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
        )

        cost_volumes_test.attrs["measure"] = "zncc"
        cost_volumes_test.attrs["window_size"] = 1
        cost_volumes_test.attrs["type_measure"] = "max"

        gt_delta_row = np.array(
            ([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
            dtype=np.float64,
        )

        gt_delta_col = np.array(
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            dtype=np.float64,
        )

        dataset_disp_map = common.dataset_disp_maps(gt_delta_row, gt_delta_col)

        test = refinement.AbstractRefinement(**{"refinement_method": "interpolation"}) # type: ignore
        delta_x, delta_y = test.refinement_method(cost_volumes_test, dataset_disp_map)

        np.testing.assert_allclose(gt_delta_row, delta_y, rtol=1e-06)
        np.testing.assert_allclose(gt_delta_col, delta_x, rtol=1e-06)

    @staticmethod
    def test_warped_image():
        """
        test warped image
        """

        ground_truth = np.array([[13, 14,  -10000,  -10000,  -10000],
                          [18, 19,  -10000,  -10000,  -10000],
                          [23, 24,  -10000,  -10000,  -10000],
                          [-10000,  -10000,  -10000,  -10000,  -10000],
                          [-10000,  -10000,  -10000,  -10000,  -10000]])

        data = np.arange(25).reshape(5, 5)
        mask = np.array(([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16)
        img = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        img.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        # all pixels are shift by 2 in row
        gt_delta_row = 2 * np.ones((5, 5))
        # all pixels are shift by 3 in column
        gt_delta_col = 3 * np.ones((5, 5))
        dataset_disp_map = common.dataset_disp_maps(gt_delta_row, gt_delta_col)

        test = refinement.AbstractRefinement(**{"refinement_method": "optical_flow"})  # type: ignore
        test_img_shift = test.warped_img(img, dataset_disp_map)

        # check that the generated image is equal to ground truth
        np.testing.assert_allclose(test_img_shift, ground_truth, atol=1e-06)

    def test_optical_flow_refinement(self):
        """
        test optical flow
        """

        dataset_disp_map = common.dataset_disp_maps(np.zeros((3, 3)), np.zeros((3, 3)))

        disparity_range_col = np.arange(-2, 2 + 1)
        disparity_range_row = np.arange(-2, 2 + 1)

        cost_volumes = xr.Dataset(
            coords={"disp_col": disparity_range_col, "disp_row": disparity_range_row}
        )
        cost_volumes.attrs["window_size"] = 3

        test = refinement.AbstractRefinement(**{"refinement_method": "optical_flow"})  # type: ignore
        map_col, map_row = test.refinement_method(cost_volumes, dataset_disp_map, self.left, self.right)

        ground_truth_row = np.array([[0, 0, 0], [0, 0.68201902, 0], [0, 0, 0]])
        ground_truth_col = np.array([[0, 0, 0], [0, -2, 0], [0, 0, 0]])

        np.testing.assert_allclose(ground_truth_row, map_row, atol=1e-06)
        np.testing.assert_allclose(ground_truth_col, map_col, atol=1e-06)
