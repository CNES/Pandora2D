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
from collections.abc import Sequence
from typing import List, Dict, NamedTuple, Any

import xarray as xr
import numpy as np

import pandora.img_tools as pandora_img_tools
from scipy.ndimage import shift


class Datasets(NamedTuple):
    """NamedTuple to store left and right datasets."""

    left: xr.Dataset
    right: xr.Dataset


def create_datasets_from_inputs(input_config: Dict, roi: Dict = None) -> Datasets:
    """
    Read image and return the corresponding xarray.DataSet

    :param input_config: configuration used to create dataset.
    :type input_config: dict
    :param roi: dictionary with a roi

            "col": {"first": <value - int>, "last": <value - int>},
            "row": {"first": <value - int>, "last": <value - int>},
            "margins": [<value - int>, <value - int>, <value - int>, <value - int>]

            with margins : left, up, right, down
    :type roi: dict
    :return: Datasets
            NamedTuple with two attributes `left` and `right` each containing a
            xarray.DataSet containing the variables :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - col_disparity: 3D (disp, row, col) xarray.DataArray float32
            - row_disparity: 3D (disp, row, col) xarray.DataArray float32

    :rtype: Datasets
    """
    check_disparities(input_config)
    return Datasets(
        pandora_img_tools.create_dataset_from_inputs(input_config["left"], roi).pipe(
            add_left_disparity_grid, input_config
        ),
        pandora_img_tools.create_dataset_from_inputs(input_config["right"], roi).pipe(
            add_right_disparity_grid, input_config
        ),
    )


def check_disparities(input_config: Dict) -> None:
    """
    Do various check against disparities properties.

    :param input_config: configuration used to create dataset.
    :type input_config:

    :raises SystemExit: If any check fails.
    """
    check_disparity_presence(input_config)
    for disparity in [input_config["col_disparity"], input_config["row_disparity"]]:
        check_disparity_types(disparity)
        check_min_max_disparity(disparity)


def check_disparity_presence(input_config):
    """
    Check that disparity keys are not missing from input_config.

    :param input_config: configuration used to create dataset.
    :type input_config:

    :raises SystemExit: if one or both keys are missing
    """
    missing = {"col_disparity", "row_disparity"} - set(input_config)
    if len(missing) == 1:
        raise KeyError(f"`{missing.pop()}` is mandatory.")
    if len(missing) == 2:
        raise KeyError("`col_disparity` and `row_disparity` are mandatory.")


def check_disparity_types(disparity: Any) -> None:
    """
    Check that disparity is a Sequence of length 2.
    :param disparity: disparity to check
    :type disparity: Any

    :raises SystemExit: if it does not meet requirements
    """
    if disparity is None or not isinstance(disparity, Sequence) or len(disparity) != 2:
        raise ValueError("Disparity should be iterable of length 2", disparity)


def check_min_max_disparity(disparity: List[int]) -> None:
    """
    Check that min disparity is lower than max disparity.

    :param disparity: disparity to check
    :type disparity: List[int]

    :raises SystemExit: if min > max
    """
    if disparity[0] > disparity[1]:
        raise ValueError(f"Min disparity ({disparity[0]}) should be lower than Max disparity ({disparity[1]})")


def add_left_disparity_grid(dataset: xr.Dataset, configuration: Dict) -> xr.Dataset:
    """
    Add left disparity to dataset.

    :param dataset: dataset to add disparity grid to
    :type dataset: xr.Dataset
    :param configuration: configuration with information about disparity
    :type configuration: Dict
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    col_disparity = configuration["col_disparity"]
    row_disparity = configuration["row_disparity"]
    return add_disparity_grid(dataset, col_disparity, row_disparity)


def add_right_disparity_grid(dataset: xr.Dataset, configuration: Dict) -> xr.Dataset:
    """
    Add right disparity to dataset.

    :param dataset: dataset to add disparity grid to
    :type dataset: xr.Dataset
    :param configuration: configuration with information about disparity
    :type configuration: Dict
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    col_disparity = sorted(-1 * value for value in configuration["col_disparity"])
    row_disparity = sorted(-1 * value for value in configuration["row_disparity"])
    return add_disparity_grid(dataset, col_disparity, row_disparity)


def add_disparity_grid(dataset: xr.Dataset, col_disparity: List[int], row_disparity: List[int]) -> xr.Dataset:
    """
    Add disparity to dataset

    :param dataset: xarray dataset
    :type dataset: xr.Dataset
    :param col_disparity: Disparity interval for columns
    :type col_disparity: List of ints
    :param row_disparity: Disparity interval for rows
    :type row_disparity: List of ints

    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    shape = (dataset.sizes["row"], dataset.sizes["col"])
    for key, disparity_interval in zip(["col_disparity", "row_disparity"], [col_disparity, row_disparity]):
        dataset[key] = xr.DataArray(
            np.array([np.full(shape, disparity) for disparity in disparity_interval]),
            dims=["band_disp", "row", "col"],
            coords={"band_disp": ["min", "max"]},
        )
        dataset.attrs[f"{key}_source"] = disparity_interval
    return dataset


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


def get_roi_processing(roi: dict, col_disparity: List[int], row_disparity: List[int]) -> dict:
    """
    Return a roi which takes disparities into account

    :param roi: roi in config file

        "col": {"first": <value - int>, "last": <value - int>},
        "row": {"first": <value - int>, "last": <value - int>},
        "margins": [<value - int>, <value - int>, <value - int>, <value - int>]
        with margins : left, up, right, down

    :param col_disparity: min and max disparities for columns.
    :type col_disparity: List[int]
    :param row_disparity: min and max disparities for rows.
    :type row_disparity: List[int]
    :type roi: Dict
    """
    new_roi = copy.deepcopy(roi)

    new_roi["margins"][0] = max(abs(col_disparity[0]), roi["margins"][0])
    new_roi["margins"][1] = max(abs(row_disparity[0]), roi["margins"][1])
    new_roi["margins"][2] = max(abs(col_disparity[1]), roi["margins"][2])
    new_roi["margins"][3] = max(abs(row_disparity[1]), roi["margins"][3])

    return new_roi
