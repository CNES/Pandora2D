# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the validity mask and criteria dataarray created in the cost volume step.
"""
import itertools
from typing import Union
import xarray as xr
import numpy as np
from numpy.typing import NDArray

from pandora.criteria import binary_dilation_msk
from pandora2d.constants import Criteria
from pandora2d.common import (
    set_out_of_col_disparity_range_to_other_value,
    set_out_of_row_disparity_range_to_other_value,
)


def allocate_criteria_dataarray(
    cv: xr.Dataset, value: Union[int, float, Criteria] = Criteria.VALID, data_type: Union[np.dtype, None] = None
) -> xr.DataArray:
    """
    This method creates the criteria_dataarray with the same dimensions as cost_volumes (cv).
    Initially, all points are considered valid and have the value XX.

    :param cv: cost_volumes
    :type cv: 4D xarray.Dataset
    :param value: value representing the valid criteria, by default Criteria.VALID = 0
    :type value: Union[int, float, Criteria]
    :param data_type: the desired data-type for the criteria_dataarray.
    :type data_type: Union[np.dtype, None], by default None
    :return: criteria_dataarray: 4D DataArray containing the criteria
    :rtype: criteria_dataarray: xr.DataArray
    """
    return xr.DataArray(
        np.full(cv.cost_volumes.shape, value, data_type),
        coords={"row": cv.row.data, "col": cv.col.data, "disp_col": cv.disp_col.data, "disp_row": cv.disp_row.data},
        dims=["row", "col", "disp_col", "disp_row"],
    )


def set_unprocessed_disp(
    criteria_dataarray: xr.DataArray,
    min_grid_col: NDArray[np.floating],
    max_grid_col: NDArray[np.floating],
    min_grid_row: NDArray[np.floating],
    max_grid_row: NDArray[np.floating],
):
    """
    This method sets PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED to points for disparities that will not be processed,
    based on the disparity grids provided.

    :param criteria_dataarray: 4D DataArray containing the criteria
    :type criteria_dataarray: xr.DataArray 4D
    :param min_grid_col: grid of min disparity col
    :type min_grid_col: NDArray[np.floating]
    :param max_grid_col: grid of max disparity col
    :type max_grid_col: NDArray[np.floating]
    :param min_grid_row: grid of min disparity row
    :type min_grid_row: NDArray[np.floating]
    :param max_grid_row: grid of max disparity row
    :type max_grid_row: NDArray[np.floating]
    """
    # Check col disparity
    set_out_of_col_disparity_range_to_other_value(
        criteria_dataarray, min_grid_col, max_grid_col, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
    )
    # Check row disparity
    set_out_of_row_disparity_range_to_other_value(
        criteria_dataarray, min_grid_row, max_grid_row, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
    )


def mask_border(offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria on the edges of the criteria_dataarray
    for each of the disparities.

    PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria is non-cumulative, so this method will be called last.

    :param offset: offset
    :type offset: int
    :param criteria_dataarray: 4D xarray.DataArray with all criteria
    :type criteria_dataarray: 4D xarray.DataArray
    """

    if offset > 0:

        # Raise criteria 0 on border of criteria_disp_col according to offset value
        criteria_dataarray.data[:offset, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataarray.data[-offset:, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataarray.data[:, :offset, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataarray.data[:, -offset:, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER


def mask_disparity_outside_right_image(offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE criteria for points with disparity dimension outside
    the right image

    :param offset: offset
    :type offset: int
    :param criteria_dataarray: 4D xarray.DataArray with all criteria
    :type criteria_dataarray: 4D xarray.DataArray
    """
    col_coords = criteria_dataarray.col.values
    row_coords = criteria_dataarray.row.values

    # Condition where the window is outside the image
    condition = (
        (criteria_dataarray.row + criteria_dataarray.disp_row < row_coords[0] + offset)
        | (criteria_dataarray.row + criteria_dataarray.disp_row > row_coords[-1] - offset)
        | (criteria_dataarray.col + criteria_dataarray.disp_col < col_coords[0] + offset)
        | (criteria_dataarray.col + criteria_dataarray.disp_col > col_coords[-1] - offset)
    )

    # Swapaxes to have same shape as cost_volumes and criteria_dataarray
    condition_swap = condition.data.swapaxes(1, 3).swapaxes(1, 2)

    # Update criteria dataarray
    criteria_dataarray.data[condition_swap] = (
        criteria_dataarray.data[condition_swap] | Criteria.PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    )


def mask_left_no_data(left_image: xr.Dataset, window_size: int, criteria_dataaray: xr.DataArray) -> None:
    """
    Set Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA on pixels where a no_data is present in the window around its
    position in the mask.

    :param left_image: left image with `msk` data var.
    :type left_image: xr.Dataset
    :param window_size: window size
    :type window_size: int
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    dilated_mask = binary_dilation_msk(left_image, window_size)
    criteria_dataaray.data[dilated_mask, ...] |= Criteria.PANDORA2D_MSK_PIXEL_LEFT_NODATA


def mask_right_no_data(img_right: xr.Dataset, window_size: int, criteria_dataarray: xr.DataArray) -> None:
    """
    Set Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA on pixels where a no_data is present in the window around its
    position in the mask shift by its disparity.

    :param img_right: right image with `msk` data var.
    :type img_right: xr.Dataset
    :param window_size: window size
    :type window_size: int
    :param criteria_dataarray:
    :type criteria_dataarray:
    """
    right_criteria_mask = np.full_like(img_right["msk"], Criteria.VALID, dtype=Criteria)
    right_binary_mask = binary_dilation_msk(img_right, window_size)
    right_criteria_mask[right_binary_mask] |= Criteria.PANDORA2D_MSK_PIXEL_RIGHT_NODATA
    for row_disp, col_disp in itertools.product(
        criteria_dataarray.coords["disp_row"], criteria_dataarray.coords["disp_col"]
    ):
        row_disp, col_disp = row_disp.data, col_disp.data
        # We arrange tests to avoid the slice [:0], which doesn’t work, while [0:] is fine.
        msk_row_slice = np.s_[:row_disp] if row_disp < 0 else np.s_[row_disp:]
        msk_col_slice = np.s_[:col_disp] if col_disp < 0 else np.s_[col_disp:]
        criteria_row_slice = np.s_[-row_disp:] if row_disp <= 0 else np.s_[:-row_disp]
        criteria_col_slice = np.s_[-col_disp:] if col_disp <= 0 else np.s_[:-col_disp]
        criteria_dataarray.loc[
            {
                "row": criteria_dataarray.coords["row"][criteria_row_slice],
                "col": criteria_dataarray.coords["col"][criteria_col_slice],
                "disp_col": col_disp,
                "disp_row": row_disp,
            }
        ] |= right_criteria_mask[msk_row_slice, msk_col_slice]
