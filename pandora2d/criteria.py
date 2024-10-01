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


def get_criteria_dataarray(left_image: xr.Dataset, right_image: xr.Dataset, cv: xr.Dataset) -> xr.DataArray:
    """
    This method fill the criteria dataarray with the different criteria obtained thanks to
    the methods implemented in this file
    """

    # Allocate criteria dataarray
    criteria_dataarray = allocate_criteria_dataarray(cv)

    if "msk" in left_image.data_vars:

        # Raise criteria PANDORA2D_MSK_PIXEL_LEFT_NODATA
        # for points having no data in left mask, for each disparity
        mask_left_no_data(left_image, cv.attrs["window_size"], criteria_dataarray)
        # Raise criteria PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT
        # for points having invalid in left mask, for each disparity
        mask_left_invalid(left_image, criteria_dataarray)

    if "msk" in right_image.data_vars:

        # Raise criteria PANDORA2D_MSK_PIXEL_RIGHT_NODATA
        # for points having no data in right mask according to disparity value
        mask_right_no_data(right_image, cv.attrs["window_size"], criteria_dataarray)
        # Raise criteria PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT
        # for points having invalid in right mask according to disparity value
        mask_right_invalid(right_image, criteria_dataarray)

    # Raise criteria PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE
    # for points for which window is outside right image according to disparity value
    mask_disparity_outside_right_image(cv.attrs["offset_row_col"], criteria_dataarray)

    # Raise criteria PANDORA2D_MSK_PIXEL_LEFT_BORDER
    # on the border according to offset value, for each disparity
    mask_border(cv.attrs["offset_row_col"], criteria_dataarray)

    # Get columns disparity grid
    d_min_col_grid = left_image["col_disparity"].sel(band_disp="min").data.copy()
    d_max_col_grid = left_image["col_disparity"].sel(band_disp="max").data.copy()

    # Get rows disparity grid
    d_min_row_grid = left_image["row_disparity"].sel(band_disp="min").data.copy()
    d_max_row_grid = left_image["row_disparity"].sel(band_disp="max").data.copy()

    # Put PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED
    # on points for which corresponding disparity is not processed
    set_unprocessed_disp(criteria_dataarray, d_min_col_grid, d_max_col_grid, d_min_row_grid, d_max_row_grid)

    return criteria_dataarray


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

    apply_right_criteria_mask(criteria_dataarray, right_criteria_mask)


def mask_left_invalid(left_image: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT criteria for points having
    an invalid point in the left image mask.
    A point is considered invalid if its value in the msk of the left image
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param left_image: left image with `msk` data var.
    :type left_image: xr.Dataset
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    invalid_left_mask = get_invalid_mask(left_image)

    criteria_dataarray.data[invalid_left_mask, ...] |= Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT


def mask_right_invalid(right_image: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT criteria for points having
    an invalid point in the right image mask shift by its disparity.
    A point is considered invalid if when we shift it by its disparity, the obtained value
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param right_image: right image with `msk` data var.
    :type right_image: xr.Dataset
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    right_criteria_mask = np.full_like(right_image["msk"], Criteria.VALID, dtype=Criteria)

    invalid_right_mask = get_invalid_mask(right_image)

    right_criteria_mask[invalid_right_mask] |= Criteria.PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT

    apply_right_criteria_mask(criteria_dataarray, right_criteria_mask)


def get_invalid_mask(image: xr.Dataset) -> NDArray:
    """
    Get mask for points of the image that are neither valid
    or no data.

    :param image: image with `msk` data var.
    :type image: xr.Dataset
    :return: invalid_mask: mask containing invalid points
    :rtype: invalid_mask: NDArray
    """

    invalid_mask = (image.msk.data != image.attrs["no_data_mask"]) & (image.msk.data != image.attrs["valid_pixels"])
    return invalid_mask


def apply_right_criteria_mask(criteria_dataarray: xr.DataArray, right_criteria_mask: NDArray):
    """
    This method apply right_criteria_mask array on criteria_dataarray according
    to row and column disparities.

    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    :param right_criteria_mask: mask to apply to criteria dataarray
    :type right_criteria_mask: np.NDArray
    """
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
