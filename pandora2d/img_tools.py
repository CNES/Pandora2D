#!/usr/bin/env python
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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
from typing import List, Dict, Union, NamedTuple, Any, Tuple
from math import floor
from numpy.typing import NDArray
from rasterio.windows import Window

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
        input_config["col_disparity"] = {"init": -9999, "range": 0}
        input_config["row_disparity"] = {"init": -9999, "range": 0}

    return Datasets(
        pandora_img_tools.create_dataset_from_inputs(input_config["left"], roi).pipe(
            add_disparity_grid, input_config["col_disparity"], input_config["row_disparity"]
        ),
        pandora_img_tools.create_dataset_from_inputs(input_config["right"], roi).pipe(
            add_disparity_grid, input_config["col_disparity"], input_config["row_disparity"], True
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
    Check that disparity a dictionary with keys "init" and range"
    where "init" is either:

        - an integer
        - a path to a grid with integer values

    :param disparity: disparity to check
    :type disparity: Any

    :raises SystemExit: if it does not meet requirements
    """

    # Check disparity type
    if disparity is None or not isinstance(disparity, Dict):
        raise ValueError("The input disparity must be a dictionary.")

    # Check that dictionary keys are correct
    if not set(disparity.keys()) == {"init", "range"}:
        raise ValueError("Disparity dictionary should contains keys : init and range", disparity)
    # Check that init is an integer or a path to a grid
    if not isinstance(disparity["init"], (int, str)):
        raise ValueError("Disparity init should be an integer or a path to a grid")

    # Check that range value is a postive integer
    if disparity["range"] < 0 or not isinstance(disparity["range"], int):
        raise ValueError("Disparity range should be an integer greater or equal to 0")


def add_disparity_grid(dataset: xr.Dataset, col_disparity: Dict, row_disparity: Dict, right=False):
    """
    Add disparity to dataset

    :param dataset: xarray dataset
    :type dataset: xr.Dataset
    :param col_disparity: Disparity interval for columns
    :type col_disparity: Dict
    :param row_disparity: Disparity interval for rows
    :type row_disparity: Dict
    :param right: indicates whether the disparity grid is added to the right dataset
    :type right: bool

    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """

    # Creates min and max disparity grids
    col_disp_min_max, col_disp_interval = get_min_max_disp_from_dicts(dataset, col_disparity, right)
    row_disp_min_max, row_disp_interval = get_min_max_disp_from_dicts(dataset, row_disparity, right)

    # Add disparity grids to dataset
    for key, disparity_data, source in zip(
        ["col_disparity", "row_disparity"], [col_disp_min_max, row_disp_min_max], [col_disp_interval, row_disp_interval]
    ):
        dataset[key] = xr.DataArray(
            disparity_data,
            dims=["band_disp", "row", "col"],
            coords={"band_disp": ["min", "max"]},
        )

        dataset.attrs[f"{key}_source"] = source
    return dataset


def get_min_max_disp_from_dicts(dataset: xr.Dataset, disparity: Dict, right: bool = False) -> Tuple[NDArray, List]:
    """
    Transforms input disparity dicts with constant init into min/max disparity grids

    :param dataset: xarray dataset
    :type dataset: xr.Dataset
    :param disparity: input disparity
    :type disparity: Dict
    :param right: indicates whether the disparity grid is added to the right dataset
    :type right: bool
    :return: 3D numpy array containing min/max disparity grids and list with disparity source
    :rtype: Tuple[NDArray, List]
    """
    disparity_dtype = np.float32
    # Creates min and max disparity grids if initial disparity is constant (int)
    if isinstance(disparity["init"], int):

        shape = (dataset.sizes["row"], dataset.sizes["col"])

        disp_interval = [
            disparity["init"] * pow(-1, right) - disparity["range"],
            disparity["init"] * pow(-1, right) + disparity["range"],
        ]

        disp_min_max = np.array([np.full(shape, disparity) for disparity in disp_interval], dtype=disparity_dtype)

    # Creates min and max disparity grids if initial disparities are variable (grid)
    elif isinstance(disparity["init"], str):

        # Get dataset coordinates to select correct zone of disparity grids if we are using a ROI
        rows = dataset.row.data
        cols = dataset.col.data

        window = Window(cols[0], rows[0], cols.size, rows.size)
        # Get disparity data
        disp_data = pandora_img_tools.rasterio_open(disparity["init"]).read(1, out_dtype=np.float32, window=window)

        # Use disparity data to creates min/max grids
        disp_min_max = np.array(
            [
                disp_data * pow(-1, right) - disparity["range"],
                disp_data * pow(-1, right) + disparity["range"],
            ],
            dtype=disparity_dtype,
        )

        disp_interval = [np.min(disp_min_max[0, ::]), np.max(disp_min_max[1, ::])]

    return disp_min_max, disp_interval


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


def get_margins_values(init_value: Union[int, np.ndarray], range_value: int, margins: list) -> Tuple[int, int]:
    """
    Generate the values of margins

    :param init_value: init value for disparity interval
    :type init_value: Union[int, np.ndarray]
    :param range_value: range value for disparity interval
    :type range_value: int
    :param margins: necessary value for margins
    :type margins: int
    :return: Margins value
    :rtype: Tuple[int, int]
    """

    disp_min = int(np.min(init_value)) - range_value
    disp_max = int(np.max(init_value)) + range_value

    return max(margins[0] - disp_min, 0), max(margins[1] + disp_max, 0)


def get_roi_processing(roi: dict, col_disparity: Dict, row_disparity: Dict) -> dict:
    """
    Return a roi which takes disparities into account.
    Update cfg roi with new margins.

    :param roi: roi in config file

        "col": {"first": <value - int>, "last": <value - int>},
        "row": {"first": <value - int>, "last": <value - int>},
        "margins": [<value - int>, <value - int>, <value - int>, <value - int>]
        with margins : left, up, right, down

    :param col_disparity: init and range for disparities in columns.
    :type col_disparity: Dict
    :param row_disparity: init and range for disparities in rows.
    :type row_disparity: Dict
    :type roi: Dict
    """

    new_roi = copy.deepcopy(roi)

    if isinstance(col_disparity["init"], str) and isinstance(row_disparity["init"], str):
        disparity_row_init = pandora_img_tools.rasterio_open(row_disparity["init"]).read()
        disparity_col_init = pandora_img_tools.rasterio_open(col_disparity["init"]).read()
    else:
        disparity_row_init = row_disparity["init"]
        disparity_col_init = col_disparity["init"]

    col_range = col_disparity["range"]
    row_range = row_disparity["range"]

    # for columns
    left, right = get_margins_values(disparity_col_init, col_range, [roi["margins"][0], roi["margins"][2]])

    # for rows
    up, down = get_margins_values(disparity_row_init, row_range, [roi["margins"][1], roi["margins"][3]])

    new_roi["margins"] = (left, up, right, down)

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
        "validity": (("row", "col", "criteria"), dataset["validity"].data[up:down, left:right, :]),
    }

    coords = {"row": row[up:down], "col": col[left:right], "criteria": dataset.criteria.values}

    new_dataset = xr.Dataset(data_variables, coords)

    new_dataset.attrs = dataset.attrs

    return new_dataset


def row_zoom_img(
    img: np.ndarray, ny: int, subpix: int, coords: Coordinates, ind: int, no_data: Union[int, str], order: int = 1
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
    :param order: The order of the spline interpolation, default is 1. The order has to be in the range 0-5.
    :type order: int, optional
    :return: an array that contains the shifted right images in row
    :rtype: array of xarray.Dataset
    """

    shift = 1 / subpix
    # For each index, shift the right image for subpixel precision 1/subpix*index
    data = zoom(img, ((ny * subpix - (subpix - 1)) / float(ny), 1), order=order)[ind::subpix, :]

    # Add a row full of no data at the end of data have the same shape as img
    # It enables to use Pandora's compute_cost_volume() methods,
    # which only accept left and right images of the same shape.
    data = np.pad(data, ((0, 1), (0, 0)), "constant", constant_values=no_data)

    row = np.arange(
        coords.get("row").values[0] + shift * ind, coords.get("row").values[-1] + 1, step=1
    )  # type: np.ndarray

    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": row, "col": coords.get("col")},
    )


def col_zoom_img(
    img: np.ndarray, nx: int, subpix: int, coords: Coordinates, ind: int, no_data: Union[int, str], order: int = 1
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
    :param order: The order of the spline interpolation, default is 1. The order has to be in the range 0-5.
    :type order: int, optional
    :return: an array that contains the shifted right images in col
    :rtype: array of xarray.Dataset
    """

    shift = 1 / subpix
    # For each index, shift the right image for subpixel precision 1/subpix*index
    data = zoom(img, (1, (nx * subpix - (subpix - 1)) / float(nx)), order=order)[:, ind::subpix]

    # Add a col full of no data at the end of data to have the same shape as img
    # It enables to use Pandora's compute_cost_volume() methods,
    # which only accept left and right images of the same shape.
    data = np.pad(data, ((0, 0), (0, 1)), "constant", constant_values=no_data)

    col = np.arange(
        coords.get("col").values[0] + shift * ind, coords.get("col").values[-1] + 1, step=1
    )  # type: np.ndarray
    return xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": coords.get("row"), "col": col},
    )


def shift_subpix_img(img_right: xr.Dataset, subpix: int, row: bool = True, order: int = 1) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted right images

    :param img_right: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_right: xarray.Dataset
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param row: row to shift (otherwise column)
    :type row: bool
    :param order: The order of the spline interpolation, default is 1. The order has to be in the range 0-5.
    :type order: int, optional
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
                        order,
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
                        order,
                    ).assign_attrs(img_right.attrs)
                )

    return img_right_shift


def shift_subpix_img_2d(img_right: xr.Dataset, subpix: int, order: int = 1) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted right images in rows and columns

    :param img_right: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_right: xarray.Dataset
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param column: column to shift (otherwise row)
    :type column: bool
    :param order: The order of the spline interpolation, default is 1. The order has to be in the range 0-5.
    :type order: int, optional
    :return: an array that contains the shifted right images
    :rtype: array of xarray.Dataset
    """

    # Row shifted images
    img_right_shift = shift_subpix_img(img_right, subpix, row=True, order=order)
    img_right_shift_2d = []

    # Columns shifted images
    for _, img in enumerate(img_right_shift):
        img_right_shift_2d += shift_subpix_img(img, subpix, row=False, order=order)

    return img_right_shift_2d
