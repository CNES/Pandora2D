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
Test common
"""
import os
import unittest

import xarray as xr
import numpy as np

from pandora2d import common


class TestCommon(unittest.TestCase):
    """
    TestImgTools class allows to test all the methods in the img_tools function
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """
        self.row = np.ones((2, 2))
        self.col = np.ones((2, 2))

    def test_save_dataset(self):
        """
        Function for testing the dataset_save function
        """
        dataset_y = xr.Dataset(
            {"row_map": (["row", "col"], self.row)},
            coords={"row": np.arange(self.row.shape[0]), "col": np.arange(self.row.shape[1])},
        )

        dataset_x = xr.Dataset(
            {"col_map": (["row", "col"], self.col)},
            coords={"row": np.arange(self.col.shape[0]), "col": np.arange(self.col.shape[1])},
        )

        dataset = dataset_y.merge(dataset_x, join="override", compat="override")

        common.save_dataset(dataset, "./tests/res_test/")
        assert os.path.exists("./tests/res_test/")

        assert os.path.exists("./tests/res_test/columns_disparity.tif")
        assert os.path.exists("./tests/res_test/row_disparity.tif")

        os.remove("./tests/res_test/columns_disparity.tif")
        os.remove("./tests/res_test/row_disparity.tif")
        os.rmdir("./tests/res_test")

    @staticmethod
    def test_dataset_disp_maps():
        """
        Test function for create a dataset
        """

        # create a test dataset for map row
        data = np.zeros((2, 2))
        dataset_y = xr.Dataset(
            {"row_map": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        # create a test dataset for map col
        dataset_x = xr.Dataset(
            {"col_map": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        # merge two dataset into one
        dataset_test = dataset_y.merge(dataset_x, join="override", compat="override")

        # create dataset with function
        dataset_fun = common.dataset_disp_maps(data, data)

        assert dataset_fun.equals(dataset_test)
