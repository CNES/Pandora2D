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
This module contains functions allowing to save the results and the configuration of Pandora pipeline.
"""
import os
import xarray as xr
import numpy as np

from pandora.common import mkdir_p, write_data_array


def save_dataset(dataset: xr.Dataset, output: str) -> None:
    """
    Save results in the output directory

    :param dataset: Dataset which contains:

        - lines : the disparity map for the lines 2D DataArray (row, col)
        - columns : the disparity map for the columns 2D DataArray (row, col)
    :type dataset: xr.Dataset
    :param output: output directory
    :type output: string
    :return: None
    """

    # create output dir
    mkdir_p(output)

    # save disp map for lines
    write_data_array(dataset["row_map"], os.path.join(output, "row_disparity.tif"))

    # save disp map for columns
    write_data_array(dataset["col_map"], os.path.join(output, "columns_disparity.tif"))


def dataset_disp_maps(delta_row: np.array, delta_col: np.array) -> xr.Dataset:
    """
    Create the dataset containing disparity maps

    :param delta_row: disparity map for row
    :type delta_row: np.array
    :param delta_col: disparity map for col
    :type delta_col: np.array
    :return: dataset: Dataset with the disparity maps with the data variables :

            - row_map 2D xarray.DataArray (row, col)
            - col_map 2D xarray.DataArray (row, col)
    :rtype: xarray.Dataset
    """

    # create a test dataset for map row
    dataset_row = xr.Dataset(
        {"row_map": (["row", "col"], delta_row)},
        coords={"row": np.arange(delta_row.shape[0]), "col": np.arange(delta_row.shape[1])},
    )
    # create a test dataset for map col
    dataset_col = xr.Dataset(
        {"col_map": (["row", "col"], delta_col)},
        coords={"row": np.arange(delta_col.shape[0]), "col": np.arange(delta_col.shape[1])},
    )
    # merge two dataset into one
    dataset = dataset_row.merge(dataset_col, join="override", compat="override")

    return dataset
