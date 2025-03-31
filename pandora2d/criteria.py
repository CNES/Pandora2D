# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
from enum import IntFlag
from typing import Union, Type, Tuple
import xarray as xr
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from pandora.criteria import binary_dilation_msk
from pandora2d.constants import Criteria
from pandora2d.common import (
    set_out_of_col_disparity_range_to_other_value,
    set_out_of_row_disparity_range_to_other_value,
)


class FlagArray(np.ndarray):
    """NDArray subclass that expects to be filled with Flags and with dedicated repr."""

    def __new__(cls, input_array: ArrayLike, flag: Type[IntFlag], dtype: DTypeLike = np.uint8):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        # add the new attribute to the created instance
        obj.flag = flag
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.flag = getattr(obj, "flag", None)  # pylint: disable=attribute-defined-outside-init

    def __repr__(self):
        if self.flag is None:
            return super().__repr__()
        max_line_width = np.get_printoptions()["linewidth"]

        flag_reprs = [repr(self.flag(i)).replace(self.flag.__name__ + ".", "") for i in range(sum(self.flag))]
        prefix = f"{self.__class__.__name__}<{self.flag.__name__}>"
        suffix = f"dtype={self.dtype}"
        array_repr = np.array2string(
            self,
            prefix=prefix,
            formatter={"int_kind": lambda x: flag_reprs[x]},
            separator=", ",
            suffix=suffix,
            max_line_width=max_line_width,
        )
        return f"{prefix}({array_repr}, {suffix})"


def get_disparity_grids(
    left_image: xr.Dataset, cv_coords: Tuple[NDArray, NDArray]
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return disparity grid from left image according to cost_volumes row and col coordinates.
    We need to use the cost volume coordinates to process the right points when the step is different from 1.

    :param left_image: left image
    :type left_image: xr.Dataset
    :param cv_coords: cost volumes row and column coordinates
    :type cv_coords: Tuple[NDArray, NDArray]
    :return: 4 disparity grids
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """

    # Get rows disparity grid
    d_min_row_grid = left_image["row_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="min").data
    d_max_row_grid = left_image["row_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="max").data

    # Get columns disparity grid
    d_min_col_grid = left_image["col_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="min").data
    d_max_col_grid = left_image["col_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="max").data

    return d_min_row_grid, d_max_row_grid, d_min_col_grid, d_max_col_grid


def allocate_criteria_dataarray(
    cv: xr.Dataset, value: Union[int, float, Criteria] = Criteria.VALID, data_type: Union[DTypeLike, None] = None
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
        FlagArray(np.full(cv.cost_volumes.shape, value, data_type), Criteria, data_type),
        coords=cv["cost_volumes"].coords,
        dims=cv["cost_volumes"].dims,
        name="criteria",
    )


def get_criteria_dataarray(left_image: xr.Dataset, right_image: xr.Dataset, cv: xr.Dataset) -> xr.DataArray:
    """
    This method fill the criteria dataarray with the different criteria obtained thanks to
    the methods implemented in this file

    :param left_image: left image
    :type left_image: xr.Dataset
    :param right_image: right image
    :type right_image: xr.Dataset
    :param cv: cost_volumes
    :type cv: 4D xarray.Dataset
    :return: criteria_dataarray: 4D DataArray containing the criteria
    :rtype: criteria_dataarray: xr.DataArray
    """

    # Allocate criteria dataarray
    criteria_dataarray = allocate_criteria_dataarray(cv, data_type=np.uint8)

    if "msk" in left_image.data_vars:

        # Raise criteria P2D_LEFT_NODATA
        # for points having no data in left mask, for each disparity
        mask_left_no_data(left_image, cv.attrs["window_size"], criteria_dataarray)
        # Raise criteria P2D_INVALID_MASK_LEFT
        # for points having invalid in left mask, for each disparity
        mask_left_invalid(left_image, criteria_dataarray)

    if "msk" in right_image.data_vars:

        # Raise criteria P2D_RIGHT_NODATA
        # for points having no data in right mask according to disparity value
        mask_right_no_data(right_image, cv.attrs["window_size"], criteria_dataarray)
        # Raise criteria P2D_INVALID_MASK_RIGHT
        # for points having invalid in right mask according to disparity value
        mask_right_invalid(right_image, criteria_dataarray)

    # Raise criteria P2D_RIGHT_DISPARITY_OUTSIDE
    # for points for which window is outside right image according to disparity value
    mask_disparity_outside_right_image(cv.attrs["offset_row_col"], criteria_dataarray)

    # Get disparity grids according to cost volumes coordinates
    d_min_row_grid, d_max_row_grid, d_min_col_grid, d_max_col_grid = get_disparity_grids(
        left_image, (cv.row.values, cv.col.values)
    )

    # Put P2D_DISPARITY_UNPROCESSED
    # on points for which corresponding disparity is not processed
    set_unprocessed_disp(criteria_dataarray, d_min_col_grid, d_max_col_grid, d_min_row_grid, d_max_row_grid)

    # Raise criteria P2D_LEFT_BORDER
    # on the border according to offset value, for each disparity
    mask_border(left_image, cv.attrs["offset_row_col"], criteria_dataarray)

    return criteria_dataarray


def set_unprocessed_disp(
    criteria_dataarray: xr.DataArray,
    min_grid_col: NDArray[np.floating],
    max_grid_col: NDArray[np.floating],
    min_grid_row: NDArray[np.floating],
    max_grid_row: NDArray[np.floating],
):
    """
    This method sets P2D_DISPARITY_UNPROCESSED to points for disparities that will not be processed,
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
        criteria_dataarray, min_grid_col, max_grid_col, Criteria.P2D_DISPARITY_UNPROCESSED
    )
    # Check row disparity
    set_out_of_row_disparity_range_to_other_value(
        criteria_dataarray, min_grid_row, max_grid_row, Criteria.P2D_DISPARITY_UNPROCESSED
    )


def mask_border(left_image: xr.Dataset, offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_LEFT_BORDER criteria on the edges of the left image
    for each of the disparities.

    P2D_LEFT_BORDER criteria is non-cumulative, so this method will be called last.

    :param left_image: left image
    :type left_image: xr.Dataset
    :param offset: offset
    :type offset: int
    :param criteria_dataarray: 4D xarray.DataArray with all criteria
    :type criteria_dataarray: 4D xarray.DataArray
    """

    left_image_border = xr.full_like(left_image["im"], 0, dtype=np.uint8)

    if offset > 0:

        left_image_border.data[:offset, :] = 1
        left_image_border.data[-offset:, :] = 1
        left_image_border.data[:, :offset] = 1
        left_image_border.data[:, -offset:] = 1

    mask = left_image_border.sel(row=criteria_dataarray.row, col=criteria_dataarray.col) == 1

    # Raise criteria 0 on border of left_image_border in criteria_dataarray according to offset value
    criteria_dataarray.data[mask] = Criteria.P2D_LEFT_BORDER


def mask_disparity_outside_right_image(offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_RIGHT_DISPARITY_OUTSIDE criteria for points with disparity dimension outside
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
    condition_swap = condition.data.swapaxes(1, 2)

    # Update criteria dataarray
    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    criteria_dataarray.data[condition_swap] |= np.uint8(Criteria.P2D_RIGHT_DISPARITY_OUTSIDE)


def mask_left_no_data(left_image: xr.Dataset, window_size: int, criteria_dataaray: xr.DataArray) -> None:
    """
    Set Criteria.P2D_LEFT_NODATA on pixels where a no_data is present in the window around its
    position in the mask.

    :param left_image: left image with `msk` data var.
    :type left_image: xr.Dataset
    :param window_size: window size
    :type window_size: int
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    dilated_mask = binary_dilation_msk(left_image, window_size)
    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    criteria_dataaray.data[dilated_mask, ...] |= np.uint8(Criteria.P2D_LEFT_NODATA)


def mask_right_no_data(img_right: xr.Dataset, window_size: int, criteria_dataarray: xr.DataArray) -> None:
    """
    Set Criteria.P2D_RIGHT_NODATA on pixels where a no_data is present in the window around its
    position in the mask shift by its disparity.

    :param img_right: right image with `msk` data var.
    :type img_right: xr.Dataset
    :param window_size: window size
    :type window_size: int
    :param criteria_dataarray:
    :type criteria_dataarray:
    """
    right_criteria_mask = np.full_like(img_right["msk"], Criteria.VALID, dtype=np.uint8)
    right_binary_mask = binary_dilation_msk(img_right, window_size)
    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    right_criteria_mask[right_binary_mask] |= np.uint8(Criteria.P2D_RIGHT_NODATA)

    apply_right_criteria_mask(criteria_dataarray, right_criteria_mask)


def mask_left_invalid(left_image: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_INVALID_MASK_LEFT criteria for points having
    an invalid point in the left image mask.
    A point is considered invalid if its value in the msk of the left image
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param left_image: left image with `msk` data var.
    :type left_image: xr.Dataset
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    invalid_left_mask = get_invalid_mask(left_image)

    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    criteria_dataarray.data[invalid_left_mask, ...] |= np.uint8(Criteria.P2D_INVALID_MASK_LEFT)


def mask_right_invalid(right_image: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_INVALID_MASK_RIGHT criteria for points having
    an invalid point in the right image mask shift by its disparity.
    A point is considered invalid if when we shift it by its disparity, the obtained value
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param right_image: right image with `msk` data var.
    :type right_image: xr.Dataset
    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    """
    right_criteria_mask = np.full_like(right_image["msk"], Criteria.VALID, dtype=np.uint8)

    invalid_right_mask = get_invalid_mask(right_image)

    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    right_criteria_mask[invalid_right_mask] |= np.uint8(Criteria.P2D_INVALID_MASK_RIGHT)

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

        # If the subpix is different from 1 in the matching cost step, we have float disparities.
        # For the moment, to decide whether to apply the P2D_RIGHT_NODATA
        # and P2D_INVALID_MASK_RIGHT criteria to subpixel disparities,
        # we use the nearest neighbor method, i.e. we apply the same criteria as to the nearest integer disparity.

        # In a future issue, we will change this method to raise these criteria if one of the points used
        # to interpolate the subpixel point has a no_data or invalid in the right mask.
        row_dsp_int, col_dsp_int = int(np.round(row_disp)), int(np.round(col_disp))

        # We arrange tests to avoid the slice [:0], which doesn’t work, while [0:] is fine.
        msk_row_slice = np.s_[:row_dsp_int] if row_dsp_int < 0 else np.s_[row_dsp_int:]  # type: ignore[index]
        msk_col_slice = np.s_[:col_dsp_int] if col_dsp_int < 0 else np.s_[col_dsp_int:]  # type: ignore[index]

        criteria_row_slice = np.s_[-row_dsp_int:] if row_dsp_int <= 0 else np.s_[:-row_dsp_int]  # type: ignore[index]
        criteria_col_slice = np.s_[-col_dsp_int:] if col_dsp_int <= 0 else np.s_[:-col_dsp_int]  # type: ignore[index]

        criteria_dataarray.loc[
            {
                "row": criteria_dataarray.coords["row"][criteria_row_slice],
                "col": criteria_dataarray.coords["col"][criteria_col_slice],
                "disp_col": col_disp,
                "disp_row": row_disp,
            }
        ] |= right_criteria_mask[msk_row_slice, msk_col_slice]


def apply_peak_on_edge(
    criteria_dataarray: xr.DataArray,
    left_image: xr.Dataset,
    cv_coords: Tuple[NDArray, NDArray],
    row_map: NDArray,
    col_map: NDArray,
):
    """
    This method raises P2D_PEAK_ON_EDGE criteria for points (row, col)
    for which the best matching cost is found for the edge of the disparity range.
    This criteria is applied on point (row, col), for each disparity value.

    :param criteria_dataaray: criteria dataarray to update
    :type criteria_dataaray: xr.DataArray
    :param left_image: left image
    :type left_image: xr.Dataset
    :param cv_coords: cost volumes row and column coordinates
    :type cv_coords: Tuple[NDArray, NDArray]
    :param row_map: row disparity map
    :type row_map: NDArray
    :param col_map: col disparity map
    :type col_map: NDArray
    """

    # Get disparity grids according to cost volumes coordinates
    d_min_row_grid, d_max_row_grid, d_min_col_grid, d_max_col_grid = get_disparity_grids(left_image, cv_coords)

    # Apply P2D_PEAK_ON_EDGE criteria
    criteria_dataarray.data[(row_map == d_min_row_grid) | (row_map == d_max_row_grid)] |= np.uint8(
        Criteria.P2D_PEAK_ON_EDGE
    )
    criteria_dataarray.data[(col_map == d_min_col_grid) | (col_map == d_max_col_grid)] |= np.uint8(
        Criteria.P2D_PEAK_ON_EDGE
    )


def allocate_validity_dataset(criteria_dataarray: xr.DataArray) -> xr.Dataset:
    """
    Allocate the validity dataset which contains an additional 'criteria' dimension.

    :param criteria_dataarray: criteria_dataarray used to create validity mask
    :type criteria_dataarray: xr.DataArray
    """

    # Get criteria names to stock them in the 'criteria' coordinate in the allocated xr.Dataset
    # We use every Criteria except the first one which corresponds to valid points.
    criteria_names = ["validity_mask"] + list(Criteria.__members__.keys())[1:]

    # In a future issue, we will change the list of names of the 'criteria' coordinate
    # to get automatically the criteria names described in constants.py
    coords = {
        "row": criteria_dataarray.coords.get("row"),
        "col": criteria_dataarray.coords.get("col"),
        "criteria": criteria_names,
    }

    dims = ("row", "col", "criteria")
    shape = (len(coords["row"]), len(coords["col"]), len(coords["criteria"]))

    # Initalize validity dataset data with zeros
    empty_data = np.full(shape, 0, dtype=np.uint8)

    dataset = xr.Dataset({"validity": xr.DataArray(empty_data, dims=dims, coords=coords)})

    return dataset


def get_validity_dataset(criteria_dataarray: xr.DataArray) -> xr.Dataset:
    """
    Fill the validity dataset which contains an additional 'criteria' dimension.

    :param criteria_dataarray: criteria_dataarray used to create validity mask
    :type criteria_dataarray: xr.DataArray
    :return: validity Dataset
    :rtype: xr.Dataset
    """

    validity_dataset = allocate_validity_dataset(criteria_dataarray)

    validity_dataset["validity"].data[:, :, 0] = get_validity_mask_band(criteria_dataarray)

    # The P2D_LEFT_BORDER criteria doesn't depend on disparities,
    # so we can use criteria_datarray at the first couple of disparities
    # to identify the points where the criteria is raised.
    validity_dataset["validity"].data[:, :, 1] = Criteria.P2D_LEFT_BORDER.is_in(criteria_dataarray[:, :, 0, 0].data)

    return validity_dataset


def get_validity_mask_band(criteria_dataarray: xr.DataArray) -> NDArray:
    """
    This method fills the validity mask band according to the criteria dataarray given as a parameter.

    This validity mask shows which points of the image are valid and which are not:
        - If a point = 2 in the validity mask band --> The point is invalid, no disparity range can be calculated
        - If a point = 1 in the validity mask band --> The point is partially valid, not all disparity range requested
          by the user have been computed
        - If a point = 0 in the validity mask band --> The point is valid, all the disparity range requested
          by the user have been computed

    :param criteria_dataarray: 4D DataArray containing the criteria
    :type criteria_dataarray: xr.DataArray
    :return: validity mask band
    :rtype: xr.DataArray
    """

    disp_range_total = len(criteria_dataarray.disp_row) * len(criteria_dataarray.disp_col)
    invalid_mask = np.full((criteria_dataarray.sizes["row"], criteria_dataarray.sizes["col"]), 0)

    for disp_col in criteria_dataarray.disp_col:
        for disp_row in criteria_dataarray.disp_row:

            # For each point, +1 is added to invalid_mask for each invalid disparity range
            invalid_mask += criteria_dataarray.sel(disp_row=disp_row, disp_col=disp_col).data != Criteria.VALID

    validity_mask = np.zeros_like(invalid_mask, dtype=np.uint8)
    # If all the disparity ranges are invalid, the point is set to 2 in the validity mask
    validity_mask[invalid_mask == disp_range_total] = 2
    # If at least one of the disparity range is invalid, the point is set to 1 in the validity mask
    validity_mask[(invalid_mask > 0) & (invalid_mask < disp_range_total)] = 1

    return validity_mask
