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
This module contains functions allowing to save the results and the configuration of Pandora pipeline.
"""
# mypy: disable-error-code="attr-defined, no-redef"
# pylint: disable=useless-import-alias

# xarray.Coordinates corresponds to the latest version of xarray.
# xarray.Coordinate corresponds to the version installed by the artifactory.
# Try/except block to be deleted once the version of xarray has been updated by CNES.
try:
    from xarray import Coordinates as Coordinates
except ImportError:
    from xarray import Coordinate as Coordinates

import os
from typing import Dict, Union, Tuple, List
import xarray as xr
import numpy as np
from numpy.typing import NDArray

from rasterio import Affine

from pandora.common import mkdir_p, write_data_array
from pandora2d.img_tools import remove_roi_margins
from pandora2d.constants import Criteria


def save_dataset(dataset: xr.Dataset, cfg: Dict, output: str) -> None:
    """
    Save results in the output directory

    :param dataset: Dataset which contains:

        - lines : the disparity map for the lines 2D DataArray (row, col)
        - columns : the disparity map for the columns 2D DataArray (row, col)
    :type dataset: xr.Dataset
    :param cfg: user configuration
    :type cfg: Dict
    :param output: output directory
    :type output: string
    :return: None
    """

    # remove ROI margins to save only user ROI in tif files
    if "ROI" in cfg:
        dataset = remove_roi_margins(dataset, cfg)
    if dataset.attrs["transform"] is not None:
        adjust_georeferencement(dataset, cfg)
    # create output dir
    mkdir_p(output)

    # save disp map for row
    write_data_array(
        dataset["row_map"],
        os.path.join(output, "row_disparity.tif"),
        crs=dataset.attrs["crs"],
        transform=dataset.attrs["transform"],
    )

    # save disp map for columns
    write_data_array(
        dataset["col_map"],
        os.path.join(output, "columns_disparity.tif"),
        crs=dataset.attrs["crs"],
        transform=dataset.attrs["transform"],
    )

    # save correlation score
    write_data_array(
        dataset["correlation_score"],
        os.path.join(output, "correlation_score.tif"),
        crs=dataset.attrs["crs"],
        transform=dataset.attrs["transform"],
    )


def adjust_georeferencement(dataset: xr.Dataset, cfg: Dict) -> None:
    """
    Change origin in case a ROI is present and set pixel size to the matching cost step.

    :param dataset: dataset to configure.
    :type dataset: xr.Dataset
    :param cfg: configuration
    :type cfg: Dict
    """
    if "ROI" in cfg:
        # Translate georeferencement origin to ROI origin:
        dataset.attrs["transform"] *= Affine.translation(cfg["ROI"]["col"]["first"], cfg["ROI"]["row"]["first"])
    row_step, col_step = get_step(cfg)
    set_pixel_size(dataset, row_step, col_step)


def get_step(cfg: Dict) -> Tuple[int, int]:
    """
    Get step from matching cost or retun default value.
    :param cfg: configuration
    :type cfg: Dict
    :return: row_step, col_step
    :rtype: Tuple[int, int]
    """
    try:
        return cfg["pipeline"]["matching_cost"]["step"]
    except KeyError:
        return 1, 1


def set_pixel_size(dataset: xr.Dataset, row_step: int = 1, col_step: int = 1) -> None:
    """
    Set the pixel size according to the step used in calculating the matching cost.

    This ensures that all pixels are well geo-referenced in case a step is applied.

    :param dataset: Data to save
    :type dataset: xr.Dataset
    :param row_step: step used in row
    :type row_step: int
    :param col_step: step used in column
    :type col_step: int
    """
    dataset.attrs["transform"] *= Affine.scale(col_step, row_step)


def dataset_disp_maps(
    delta_row: np.ndarray,
    delta_col: np.ndarray,
    coords: Coordinates,
    correlation_score: np.ndarray,
    attributes: dict = None,
) -> xr.Dataset:
    """
    Create the dataset containing disparity maps and score maps

    :param delta_row: disparity map for row
    :type delta_row: np.ndarray
    :param delta_col: disparity map for col
    :type delta_col: np.ndarray
    :param coords: disparity maps coordinates
    :type coords: xr.Coordinates
    :param correlation_score: score map
    :type correlation_score: np.ndarray
    :param attributes: disparity map for col
    :type attributes: dict
    :return: dataset: Dataset with the disparity maps and score with the data variables :

            - row_map 2D xarray.DataArray (row, col)
            - col_map 2D xarray.DataArray (row, col)
            - score 2D xarray.DataArray (row, col)
    :rtype: xarray.Dataset
    """

    # Raise an error if col coordinate is missing
    if coords.get("col") is None:
        raise ValueError("The col coordinate does not exist")
    # Raise an error if row coordinate is missing
    if coords.get("row") is None:
        raise ValueError("The row coordinate does not exist")

    coords = {
        "row": coords.get("row"),
        "col": coords.get("col"),
    }

    dims = ("row", "col")

    dataarray_row = xr.DataArray(delta_row, dims=dims, coords=coords)
    dataarray_col = xr.DataArray(delta_col, dims=dims, coords=coords)
    dataarray_score = xr.DataArray(correlation_score, dims=dims, coords=coords)

    dataset = xr.Dataset({"row_map": dataarray_row, "col_map": dataarray_col, "correlation_score": dataarray_score})

    if attributes is not None:
        dataset.attrs = attributes

    return dataset


def set_out_of_row_disparity_range_to_other_value(
    data: xr.DataArray,
    min_disp_grid: NDArray[np.floating],
    max_disp_grid: NDArray[np.floating],
    value: Union[int, float, Criteria],
    global_disparity_range: Union[None, List[int]] = None,
) -> None:
    """
    Put special value in data  where the disparity is out of the range defined by disparity grids.

    The operation is done inplace.

    :param data: cost_volumes or criteria_dataarray to modify.
    :type data: xr.DataArray 4D
    :param min_disp_grid: grid of min disparity.
    :type min_disp_grid: NDArray[np.floating]
    :param max_disp_grid: grid of max disparity.
    :type max_disp_grid: NDArray[np.floating]
    :param value: value to set on data.
    :type value: Union[int, float, Criteria]
    :param global_disparity_range:
    :type global_disparity_range:
    """
    # WARNING: if one day we switch disp_row with disp_col index should be -2
    ndisp_row = data.shape[-1]

    # We want to put special value on points that are not in the global disparity range (row_disparity_source)
    for disp_row in range(ndisp_row):
        if global_disparity_range is not None:  # Case we are working with cost volume
            masking = np.nonzero(
                np.logical_or(
                    (data.coords["disp_row"].data[disp_row] < min_disp_grid)
                    & (data.coords["disp_row"].data[disp_row] >= global_disparity_range[0]),
                    (data.coords["disp_row"].data[disp_row] > max_disp_grid)
                    & (data.coords["disp_row"].data[disp_row] <= global_disparity_range[1]),
                )
            )
        else:
            masking = np.nonzero(
                np.logical_or(
                    data.coords["disp_row"].data[disp_row] < min_disp_grid,
                    data.coords["disp_row"].data[disp_row] > max_disp_grid,
                )
            )
        data.data[masking[0], masking[1], :, disp_row] = value


def set_out_of_col_disparity_range_to_other_value(
    data: xr.DataArray,
    min_disp_grid: NDArray[np.floating],
    max_disp_grid: NDArray[np.floating],
    value: Union[int, float, Criteria],
    global_disparity_range: Union[None, List[int]] = None,
) -> None:
    """
    Put special value in data (cost_volumes or criteria_dataarray) where the disparity is out of the range defined
    by disparity grids.

    The operation is done inplace.

    :param data: cost_volumes or criteria_dataarray to modify.
    :type data: xr.DataArray 4D
    :param min_disp_grid: grid of min disparity.
    :type min_disp_grid: NDArray[np.floating]
    :param max_disp_grid: grid of max disparity.
    :type max_disp_grid: NDArray[np.floating]
    :param value: value to set on data.
    :type value: Union[int, float, Criteria]
    :param global_disparity_range:
    :type global_disparity_range:
    """
    # WARNING: if one day we switch disp_row with disp_col index should be -1
    ndisp_col = data.shape[-2]

    # We want to put special value on points that are not in the global disparity range (col_disparity_source)
    for disp_col in range(ndisp_col):
        if global_disparity_range is not None:  # Case we are working with cost volume
            masking = np.nonzero(
                np.logical_or(
                    (data.coords["disp_col"].data[disp_col] < min_disp_grid)
                    & (data.coords["disp_col"].data[disp_col] >= global_disparity_range[0]),
                    (data.coords["disp_col"].data[disp_col] > max_disp_grid)
                    & (data.coords["disp_col"].data[disp_col] <= global_disparity_range[1]),
                )
            )
        else:
            masking = np.nonzero(
                np.logical_or(
                    data.coords["disp_col"].data[disp_col] < min_disp_grid,
                    data.coords["disp_col"].data[disp_col] > max_disp_grid,
                )
            )
        data.data[masking[0], masking[1], disp_col, :] = value
