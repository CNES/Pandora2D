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

from pandora.common import mkdir_p, write_data_array


def save_dataset(dataset: xr.Dataset, output: str) -> None:
    """
     Save results in the output directory

    Args:
        dataset (xr.Dataset): Dataset which contains:

        - lines : the disparity map for the lines 2D DataArray (row, col)
        - columns : the disparity map for the columns 2D DataArray (row, col)

        output (str): output directory
    """

    # create output dir
    mkdir_p(output)

    # save disp map for lines
    write_data_array(dataset["im_line"], os.path.join(output, "lines_diparity.tif"))

    # save disp map for columns
    write_data_array(dataset["im_col"], os.path.join(output, "columns_diparity.tif"))
