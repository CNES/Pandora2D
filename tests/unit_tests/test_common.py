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
Test common
"""

# pylint: disable=redefined-outer-name
import os

import numpy as np
import pytest
import xarray as xr

from pandora2d import common


@pytest.fixture
def create_test_dataset():
    """
    Create a test dataset
    """
    row, col = np.ones((2, 2)), np.ones((2, 2))

    dataset_y = xr.Dataset(
        {"row_map": (["row", "col"], row)},
        coords={"row": np.arange(row.shape[0]), "col": np.arange(row.shape[1])},
    )

    dataset_x = xr.Dataset(
        {"col_map": (["row", "col"], col)},
        coords={"row": np.arange(col.shape[0]), "col": np.arange(col.shape[1])},
    )

    dataset = dataset_y.merge(dataset_x, join="override", compat="override")

    return dataset


def test_save_dataset(create_test_dataset):
    """
    Function for testing the dataset_save function
    """

    common.save_dataset(create_test_dataset, "./tests/res_test/")
    assert os.path.exists("./tests/res_test/")

    assert os.path.exists("./tests/res_test/columns_disparity.tif")
    assert os.path.exists("./tests/res_test/row_disparity.tif")

    os.remove("./tests/res_test/columns_disparity.tif")
    os.remove("./tests/res_test/row_disparity.tif")
    os.rmdir("./tests/res_test")


def test_dataset_disp_maps(create_test_dataset):
    """
    Test function for create a dataset
    """

    dataset_test = create_test_dataset

    dataset_test.attrs = {"invalid_disp": -9999}

    # create dataset with function
    dataset_fun = common.dataset_disp_maps(np.ones((2, 2)), np.ones((2, 2)), {"invalid_disp": -9999})

    assert dataset_fun.equals(dataset_test)
