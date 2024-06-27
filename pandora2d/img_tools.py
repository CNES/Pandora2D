#!/usr/bin/env python
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
This module contains functions associated to raster images.
"""

# mypy: disable-error-code="attr-defined, no-redef"
# pylint: disable="useless-import-alias,redefined-outer-name"
# xarray.Coordinates corresponds to the latest version of xarray.
# xarray.Coordinate corresponds to the version installed by the artifactory.
# Try/except block to be deleted once the version of xarray has been updated by CNES.
try:
    from xarray import Coordinates as Coordinates
except ImportError:
    from xarray import Coordinate as Coordinates

import copy
from collections.abc import Sequence
from typing import List, Dict, Union, NamedTuple, Any

from math import floor
import xarray as xr
import numpy as np
from scipy.ndimage import shift, zoom

import pandora.img_tools as pandora_img_tools


class Datasets(NamedTuple):
    """NamedTuple to store left and right datasets."""

    left: xr.Dataset
    right: xr.Dataset


def create_datasets_from_inputs(input_config: Dict, roi: Dict = None, estimation_cfg: Dict = None) -> Datasets:
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
    :param estimation_cfg: dictionary containing estimation configuration
    :type estimation_cfg: dict
    :return: Datasets
            NamedTuple with two attributes `left` and `right` each containing a
            xarray.DataSet containing the variables :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - col_disparity: 3D (disp, row, col) xarray.DataArray float32
            - row_disparity: 3D (disp, row, col) xarray.DataArray float32

    :rtype: Datasets
    """
    if estimation_cfg is None:
        check_disparities(input_config)
    else:
        input_config["col_disparity"] = [-9999, -9999]
        input_config["row_disparity"] = [-9999, -9999]

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


def shift_disp_row_img(img_right: xr.Dataset, dec_row: int) -> xr.Dataset:
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
    # coordinates of images
    row = img_right.get("row")
    col = img_right.get("col")

    # shifted image by scipy
    data = shift(img_right["im"].data, (-dec_row, 0), cval=img_right.attrs["no_data_img"])
    # create shifted image dataset
    img_right_shift = xr.Dataset({"im": (["row", "col"], data)}, coords={"row": row, "col": col})
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
    Return a roi which takes disparities into account.
    Update cfg roi with new margins.

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

    new_roi["margins"] = (
        max(abs(col_disparity[0]), roi["margins"][0]),
        max(abs(row_disparity[0]), roi["margins"][1]),
        max(abs(col_disparity[1]), roi["margins"][2]),
        max(abs(row_disparity[1]), roi["margins"][3]),
    )

    # Update user ROI with new margins.
    roi["margins"] = new_roi["margins"]

    return new_roi


def remove_roi_margins(dataset: xr.Dataset, cfg: Dict):
    """
    Remove ROI margins before saving output dataset

    :param dataset: dataset containing disparity row and col maps
    :type dataset: xr.Dataset
    :param cfg: output configuration of the pandora2d machine
    :type cfg: Dict
    """

    step = cfg["pipeline"]["matching_cost"]["step"]

    row = dataset.row.data
    col = dataset.col.data

    # Initialized indexes to get right rows and columns
    (left, up, right, down) = (0, 0, len(col), len(row))

    # Example with col = [8,10,12,14,16],  step_col=2, row = [0,4,8,12], step_row=4
    # ROI={
    #   {"col": "first": 10, "last": 14},
    #   {"row": "first": 0, "last": 10} }
    #   {"margins": (3,3,3,3)}

    # According to ROI, we want new_col=[10,12,14]=col[1:-1]
    # with 1=floor((cfg["ROI"]["col"]["first"] - col[0]) / step_col)=left
    # and -1=floor((cfg["ROI"]["col"]["last"] - col[-1]) / step_col)=right

    # According to ROI, we want new_row=[0,4,8]=row[0:-1]
    # with 0=initialized up
    # and -1=floor((cfg["ROI"]["row"]["last"] - row[-1]) / step[0])=down

    # Get the correct indexes to get the right columns based on the user ROI
    if col[0] < cfg["ROI"]["col"]["first"]:
        left = floor((cfg["ROI"]["col"]["first"] - col[0]) / step[1])
    if col[-1] > cfg["ROI"]["col"]["last"]:
        right = floor((cfg["ROI"]["col"]["last"] - col[-1]) / step[1])

    # Get the correct indexes to get the right rows based on the user ROI
    if row[0] < cfg["ROI"]["row"]["first"]:
        up = floor((cfg["ROI"]["row"]["first"] - row[0]) / step[0])
    if row[-1] > cfg["ROI"]["row"]["last"]:
        down = floor((cfg["ROI"]["row"]["last"] - row[-1]) / step[0])

    # Create a new dataset with right rows and columns.
    data_variables = {
        "row_map": (("row", "col"), dataset["row_map"].data[up:down, left:right]),
        "col_map": (("row", "col"), dataset["col_map"].data[up:down, left:right]),
        "correlation_score": (("row", "col"), dataset["correlation_score"].data[up:down, left:right]),
    }

    coords = {"row": row[up:down], "col": col[left:right]}

    new_dataset = xr.Dataset(data_variables, coords)

    new_dataset.attrs = dataset.attrs

    return new_dataset


def row_zoom_img(
    img: np.ndarray, ny: int, subpix: int, coords: Coordinates, ind: int, no_data: Union[int, str]
) -> xr.Dataset:
    """
    Return a list that contains the shifted right images in row

    This method is temporary, the user can then choose the filter for this function

    :param img: image to shift
    :type img: np.ndarray
    :param ny: row number in data
    :type ny: int
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param coords: coordinates for output datasets
    :type coords: Coordinates
    :param ind: index of range(subpix)
    :type ind: int
    :param no_data: no_data value in img
    :type no_data: Union[int, str]
    :return: an array that contains the shifted right images in row
    :rtype: array of xarray.Dataset
    """

    shift = 1 / subpix
    # For each index, shift the right image for subpixel precision 1/subpix*index
    data = zoom(img, ((ny * subpix - (subpix - 1)) / float(ny), 1), order=1)[ind::subpix, :]

    # Add a row full of no data at the end of data have the same shape as img
    # It enables to use Pandora's compute_cost_volume() methods,
    # which only accept left and right images of the same shape.
    data = np.pad(data, ((0, 1), (0, 0)), "constant", constant_values=no_data)

    row = np.arange(coords.get("row")[0] + shift * ind, coords.get("row")[-1] + 1, step=1)  # type: np.ndarray

    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": row, "col": coords.get("col")},
    )


def col_zoom_img(
    img: np.ndarray, nx: int, subpix: int, coords: Coordinates, ind: int, no_data: Union[int, str]
) -> xr.Dataset:
    """
    Return a list that contains the shifted right images in col

    This method is temporary, the user can then choose the filter for this function

    :param img: image to shift
    :type img: np.ndarray
    :param nx: col number in data
    :type nx: int
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param coords: coordinates for output datasets
    :type coords: Coordinates
    :param ind: index of range(subpix)
    :type ind: int
    :param no_data: no_data value in img
    :type no_data: Union[int, str]
    :return: an array that contains the shifted right images in col
    :rtype: array of xarray.Dataset
    """

    shift = 1 / subpix
    # For each index, shift the right image for subpixel precision 1/subpix*index
    data = zoom(img, (1, (nx * subpix - (subpix - 1)) / float(nx)), order=1)[:, ind::subpix]

    # Add a col full of no data at the end of data to have the same shape as img
    # It enables to use Pandora's compute_cost_volume() methods,
    # which only accept left and right images of the same shape.
    data = np.pad(data, ((0, 0), (0, 1)), "constant", constant_values=no_data)

    col = np.arange(coords.get("col")[0] + shift * ind, coords.get("col")[-1] + 1, step=1)  # type: np.ndarray
    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": coords.get("row"), "col": col},
    )


def shift_subpix_img(img_right: xr.Dataset, subpix: int, row: bool = True) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted right images

    :param img_right: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_right: xarray.Dataset
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param column: column to shift (otherwise row)
    :type column: bool
    :return: an array that contains the shifted right images
    :rtype: array of xarray.Dataset
    """
    img_right_shift = [img_right]

    if subpix > 1:
        for ind in np.arange(1, subpix):
            if row:
                img_right_shift.append(
                    row_zoom_img(
                        img_right["im"].data,
                        img_right.sizes["row"],
                        subpix,
                        img_right.coords,
                        ind,
                        img_right.attrs["no_data_img"],
                    ).assign_attrs(img_right.attrs)
                )
            else:
                img_right_shift.append(
                    col_zoom_img(
                        img_right["im"].data,
                        img_right.sizes["col"],
                        subpix,
                        img_right.coords,
                        ind,
                        img_right.attrs["no_data_img"],
                    ).assign_attrs(img_right.attrs)
                )

    return img_right_shift
