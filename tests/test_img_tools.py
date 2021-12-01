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
Test configuration
"""
import unittest
import xarray as xr
import numpy as np


from pandora2d import img_tools


class TestImgTools(unittest.TestCase):
    """
    TestImgTools class allows to test all the methods in the img_tools function
    """

    def setUp(self) -> None:
        """
        Method called to prepare the test fixture

        """
        # original image
        data = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))
        # original mask
        mask = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        # create original dataset
        self.data = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        # add attributes for mask
        self.data.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,  # arbitrary default value
            "no_data_mask": 1,
        }
        no_data_pixels = np.where(data == np.nan)
        self.data["msk"] = xr.DataArray(
            np.full((data.shape[0], data.shape[1]), self.data.attrs["valid_pixels"]).astype(np.int16),
            dims=["row", "col"],
        )
        # associate nan value in mask to the no_data param
        self.data["msk"].data[no_data_pixels] = int(self.data.attrs["no_data_mask"])

        # create the dataset of an image with dec_y = 1
        shifted_data = np.array([[1, 1, 1], [1, 1, 1], [-9999, -9999, -9999]])
        # original mask
        shifted_mask = np.array(([1, 1, 1], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
        self.data_down = xr.Dataset(
            {"im": (["row", "col"], shifted_data), "msk": (["row", "col"], shifted_mask)},
            coords={"row": np.arange(shifted_data.shape[0]), "col": np.arange(shifted_data.shape[1])},
        )

        self.data_down.attrs = {
            "no_data_img": -9999,
            "valid_pixels": 0,  # arbitrary default value
            "no_data_mask": 1,
        }

        no_data_pixels = np.where(shifted_data == -9999)
        self.data_down["msk"] = xr.DataArray(
            np.full((shifted_data.shape[0], shifted_data.shape[1]), self.data_down.attrs["valid_pixels"]).astype(
                np.int16
            ),
            dims=["row", "col"],
        )
        # associate nan value in mask to the no_data param
        self.data_down["msk"].data[no_data_pixels] = int(self.data_down.attrs["no_data_mask"])

    def test_shift_img_pandora2d(self):
        """
        Test of shift_img_pandora_2d function
        """
        my_data_down = img_tools.shift_img_pandora2d(self.data, 1)
        assert my_data_down.equals(self.data_down)
