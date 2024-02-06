#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Test common
"""
# pylint: disable=redefined-outer-name

import os

import pytest
import xarray as xr
import numpy as np

from pandora2d import common


@pytest.fixture
def create_output_dataset():
    """
    Create dataset containing disparity maps and correlation score
    """

    data = np.zeros((2, 2))

    coords = {
        "row": np.arange(data.shape[0]),
        "col": np.arange(data.shape[1]),
    }

    dims = ("row", "col")

    dataarray_row = xr.DataArray(data, dims=dims, coords=coords)
    dataarray_col = xr.DataArray(data, dims=dims, coords=coords)
    dataarray_score = xr.DataArray(data, dims=dims, coords=coords)

    dataset = xr.Dataset({"row_map": dataarray_row, "col_map": dataarray_col, "correlation_score": dataarray_score})

    return dataset


def test_save_dataset(create_output_dataset):
    """
    Function for testing the dataset_save function
    """

    dataset = create_output_dataset

    common.save_dataset(dataset, "./tests/res_test/")
    assert os.path.exists("./tests/res_test/")

    assert os.path.exists("./tests/res_test/columns_disparity.tif")
    assert os.path.exists("./tests/res_test/row_disparity.tif")

    assert os.path.exists("./tests/res_test/correlation_score.tif")

    os.remove("./tests/res_test/columns_disparity.tif")
    os.remove("./tests/res_test/row_disparity.tif")
    os.remove("./tests/res_test/correlation_score.tif")
    os.rmdir("./tests/res_test")


def test_dataset_disp_maps(create_output_dataset):
    """
    Test function for create a dataset
    """

    # merge two dataset into one
    dataset_test_gt = create_output_dataset
    dataset_test_gt.attrs = {"invalid_disp": -9999}

    # create dataset with function
    data = np.zeros((2, 2))
    dataset_fun = common.dataset_disp_maps(data, data, data, {"invalid_disp": -9999})

    assert dataset_fun.equals(dataset_test_gt)
