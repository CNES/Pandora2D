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

from pandora2d import refinement, common


class TestRefinement(unittest.TestCase):
    """
    TestRefinement class allows to test the refinement module
    """

    @staticmethod
    def test_check_conf():
        """
        Test the interpolation method
        """

        refinement.AbstractRefinement(**{"refinement_method": "interpolation"}) # type: ignore

        with pytest.raises(KeyError):
            refinement.AbstractRefinement(**{"refinement_method": "wta"}) # type: ignore

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

        gt_delta_y = np.array(
            ([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
            dtype=np.float64,
        )

        gt_delta_x = np.array(
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            dtype=np.float64,
        )

        dataset_disp_map = common.dataset_disp_maps(gt_delta_y, gt_delta_x)

        test = refinement.AbstractRefinement(**{"refinement_method": "interpolation"}) # type: ignore
        delta_x, delta_y = test.refinement_method(cost_volumes_test, dataset_disp_map)

        np.testing.assert_allclose(gt_delta_y, delta_y, rtol=1e-06)
        np.testing.assert_allclose(gt_delta_x, delta_x, rtol=1e-06)
