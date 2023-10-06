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
This module contains functions associated to raster images.
"""


import copy
import xarray as xr
import numpy as np

from scipy.ndimage import shift


def shift_img_pandora2d(img_right: xr.Dataset, dec_row: int) -> xr.Dataset:
    """
    Return a Dataset that contains the shifted right images

    :param img_right: right Dataset image containing :
                - im : 2D (row, col) xarray.Datasat
    :type img_right: xr.Dataset
    :param dec_row: the value of shifting for dispy
    :type dec_row: int
    :return: img_right_shift: Dataset containing the shifted image
    :rtype: xr.Dataset
    """
    # dimensions of images
    nrow_, ncol_ = img_right["im"].shape
    # shifted image by scipy
    data = shift(img_right["im"].data, (-dec_row, 0), cval=img_right.attrs["no_data_img"])
    # create shifted image dataset
    img_right_shift = xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(nrow_), "col": np.arange(ncol_)}
    )
    # add attributes to dataset
    img_right_shift.attrs = {
        "no_data_img": img_right.attrs["no_data_img"],
        "valid_pixels": 0,  # arbitrary default value
        "no_data_mask": 1,
    }  # arbitrary default value
    # Pandora replace all nan values by -9999
    no_data_pixels = np.where(data == img_right.attrs["no_data_img"])
    # add mask to the shifted image in dataset
    img_right_shift["msk"] = xr.DataArray(
        np.full((data.shape[0], data.shape[1]), img_right_shift.attrs["valid_pixels"]).astype(np.int16),
        dims=["row", "col"],
    )
    # associate nan value in mask to the no_data param
    img_right_shift["msk"].data[no_data_pixels] = int(img_right_shift.attrs["no_data_mask"])

    return img_right_shift

def get_roi_processing(
    roi : dict,
    disp_min_col: int,
    disp_max_col: int,
    disp_min_row: int,
    disp_max_row: int) -> dict:
    """
    Return a roi which takes disparities into account

    :param roi: roi in config file

        "col": {"first": <value - int>, "last": <value - int>},
        "row": {"first": <value - int>, "last": <value - int>},
        "margins": [<value - int>, <value - int>, <value - int>, <value - int>]
        with margins : left, up, right, down

    :type roi: Dict
    :param disp_min_col: minimal disparity for columns
    :type disp_min_col: int
    :param disp_max_col: maximal disparity for columns
    :type disp_max_col: int
    :param disp_min_row: minimal disparity for lines
    :type disp_min_row: int
    :param disp_max_row: maximal disparity for lines
    :type disp_max_row: int
    """
    new_roi = copy.deepcopy(roi)

    new_roi["margins"][0] = max(abs(disp_min_col), roi["margins"][0])
    new_roi["margins"][1] = max(abs(disp_min_row), roi["margins"][1])
    new_roi["margins"][2] = max(abs(disp_max_col), roi["margins"][2])
    new_roi["margins"][3] = max(abs(disp_max_row), roi["margins"][3])

    return new_roi
