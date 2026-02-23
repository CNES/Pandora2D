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
from functools import partial

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.ndimage import binary_dilation

from pandora2d.constants import Criteria

DISPARITY_INDEPENDENT_CRITERIA = {Criteria.P2D_LEFT_BORDER, Criteria.P2D_LEFT_NODATA, Criteria.P2D_INVALID_MASK_LEFT}
DISPARITY_DEPENDENT_CRITERIA = set(Criteria) - {Criteria.VALID} - DISPARITY_INDEPENDENT_CRITERIA


class FlagArray(np.ndarray):
    """NDArray subclass that expects to be filled with Flags and with dedicated repr."""

    def __new__(cls, input_array: ArrayLike, flag: type[IntFlag], dtype: DTypeLike = np.uint8):
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

    def __repr__(self) -> str:
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
    left_image: xr.Dataset, cv_coords: tuple[NDArray, NDArray]
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Return disparity grid from left image according to cost_volumes row and col coordinates.
    We need to use the cost volume coordinates to process the right points when the step is different from 1.

    :param left_image: left image
    :param cv_coords: cost volumes row and column coordinates
    :return: 4 disparity grids
    """

    # Get rows disparity grid
    d_min_row_grid = left_image["row_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="min").data
    d_max_row_grid = left_image["row_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="max").data

    # Get columns disparity grid
    d_min_col_grid = left_image["col_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="min").data
    d_max_col_grid = left_image["col_disparity"].sel(row=cv_coords[0], col=cv_coords[1], band_disp="max").data

    return d_min_row_grid, d_max_row_grid, d_min_col_grid, d_max_col_grid


def binary_dilation_dataarray(img: xr.Dataset, window_size: int) -> xr.DataArray:
    """
    Apply scipy binary_dilation on our image dataset while keeping the coordinates.
    Get the no_data pixels.

    :param img: Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :param window_size: window size of the cost volume
    :return: np.ndarray with location of pixels that are marked as no_data according to the image mask
    """

    dilated_mask = binary_dilation(
        img["msk"].data == img.attrs["no_data_mask"],
        structure=np.ones((window_size, window_size)),
        iterations=1,
    )

    return img["msk"].copy(data=dilated_mask)


def allocate_criteria_dataarray(
    cv: xr.Dataset, value: int | float | Criteria = Criteria.VALID, data_type: DTypeLike | None = None
) -> xr.DataArray:
    """
    This method creates the criteria_dataarray with the same dimensions as cost_volumes (cv).
    Initially, all points are considered valid and have the value XX.

    :param cv: cost_volumes
    :param value: value representing the valid criteria, by default Criteria.VALID = 0
    :param data_type: the desired data-type for the criteria_dataarray.
    :return: criteria_dataarray: 4D DataArray containing the criteria
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
    :param right_image: right image
    :param cv: cost_volumes
    :return: criteria_dataarray: 4D DataArray containing the criteria
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
        mask_right_no_data(right_image, cv.attrs["window_size"], criteria_dataarray, cv.attrs["spline_order_filter"])
        # Raise criteria P2D_INVALID_MASK_RIGHT
        # for points having invalid in right mask according to disparity value
        mask_right_invalid(right_image, criteria_dataarray)

    # Raise criteria P2D_RIGHT_DISPARITY_OUTSIDE
    # for points for which window is outside right image according to disparity value
    mask_disparity_outside_right_image(right_image, cv.attrs["offset_row_col"], criteria_dataarray)

    # Raise criteria P2D_LEFT_BORDER
    # on the border according to offset value, for each disparity
    mask_border(left_image, cv.attrs["offset_row_col"], criteria_dataarray)

    return criteria_dataarray


def mask_border(left_image: xr.Dataset, offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_LEFT_BORDER criteria on the edges of the left image
    for each of the disparities.

    P2D_LEFT_BORDER criteria is non-cumulative, so this method will be called last.

    :param left_image: left image
    :param offset: offset
    :param criteria_dataarray: 4D xarray.DataArray with all criteria
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


def mask_disparity_outside_right_image(img_right: xr.Dataset, offset: int, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_RIGHT_DISPARITY_OUTSIDE criteria for points with disparity dimension outside
    the right image

    :param img_right: right image.
    :param offset: offset
    :param criteria_dataarray: 4D xarray.DataArray with all criteria
    """
    row_coords = img_right.row.values
    col_coords = img_right.col.values

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


def mask_left_no_data(left_image: xr.Dataset, window_size: int, criteria_dataarray: xr.DataArray) -> None:
    """
    Set Criteria.P2D_LEFT_NODATA on pixels where a no_data is present in the window around its
    position in the mask.

    :param left_image: left image with `msk` data var.
    :param window_size: window size
    :param criteria_dataaray: criteria dataarray to update
    """
    dilated_mask = binary_dilation_dataarray(left_image, window_size).sel(
        row=criteria_dataarray.row, col=criteria_dataarray.col
    )
    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    criteria_dataarray.data[dilated_mask, ...] |= np.uint8(Criteria.P2D_LEFT_NODATA)


def mask_right_no_data(
    img_right: xr.Dataset, window_size: int, criteria_dataarray: xr.DataArray, spline_order_filter: int
) -> None:
    """
    Set Criteria.P2D_RIGHT_NODATA on pixels where a no_data is present in the window around its
    position in the mask shift by its disparity.

    :param img_right: right image with `msk` data var.
    :param window_size: window size
    :param criteria_dataaray: criteria dataarray to update
    :param spline_order_filter: order of the scipy filter
    """
    mask_criteria_right = xr.full_like(img_right["msk"], np.uint8(Criteria.VALID), dtype=np.uint8)

    right_binary_mask = binary_dilation_dataarray(img_right, window_size)
    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    mask_criteria_right.data[right_binary_mask] |= np.uint8(Criteria.P2D_RIGHT_NODATA)

    apply_nodata_right_criteria_mask(criteria_dataarray, mask_criteria_right, spline_order_filter)


def mask_left_invalid(left_image: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_INVALID_MASK_LEFT criteria for points having
    an invalid point in the left image mask.
    A point is considered invalid if its value in the msk of the left image
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param left_image: left image with `msk` data var.
    :param criteria_dataaray: criteria dataarray to update
    """
    invalid_left_mask = get_invalid_mask(left_image).sel(row=criteria_dataarray.row, col=criteria_dataarray.col)

    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    criteria_dataarray.data[invalid_left_mask.data, ...] |= np.uint8(Criteria.P2D_INVALID_MASK_LEFT)


def mask_right_invalid(img_right: xr.Dataset, criteria_dataarray: xr.DataArray) -> None:
    """
    This method raises P2D_INVALID_MASK_RIGHT criteria for points having
    an invalid point in the right image mask shift by its disparity.
    A point is considered invalid if when we shift it by its disparity, the obtained value
    is different from the values of the valid_pixels and no_data_mask attributes.

    :param img_right: right image with `msk` data var.
    :param criteria_dataaray: criteria dataarray to update
    """

    mask_criteria_right = xr.full_like(img_right["msk"], np.uint8(Criteria.VALID), dtype=np.uint8)

    invalid_right_mask = get_invalid_mask(img_right)

    # With in place operation we need to cast Criteria (seen as int64). Seems to be related to unsigned.
    mask_criteria_right.data[invalid_right_mask.data] |= np.uint8(Criteria.P2D_INVALID_MASK_RIGHT)

    apply_invalid_right_criteria_mask(criteria_dataarray, mask_criteria_right)


def get_invalid_mask(image: xr.Dataset) -> xr.DataArray:
    """
    Get mask for points of the image that are neither valid
    or no data.

    :param image: image with `msk` data var.
    :return: invalid_mask: mask containing invalid points
    """

    invalid_mask = (image.msk != image.attrs["no_data_mask"]) & (image.msk != image.attrs["valid_pixels"])

    return invalid_mask


def apply_invalid_right_criteria_mask(criteria_dataarray: xr.DataArray, mask_criteria_right: xr.DataArray) -> None:
    """
    This method apply mask_criteria_right array on criteria_dataarray according
    to row and column disparities.
    Ignore invalid pixels from the input mask used in interpolation to raise the P2D_INVALID_MASK_RIGHT criterion

    :param criteria_dataaray: criteria dataarray to update
    :param mask_criteria_right: mask to apply to criteria dataarray
    """

    # We use a temporary mask of the same size as the image to correctly handle cases where the step is different from 1
    mask_img_shape = np.zeros_like(mask_criteria_right, dtype=np.uint8)

    p = partial(filter, lambda x: x.is_integer())

    for row_disp, col_disp in itertools.product(
        p(criteria_dataarray.coords["disp_row"].data), p(criteria_dataarray.coords["disp_col"].data)
    ):

        row_dsp_int, col_dsp_int = int(row_disp), int(col_disp)
        # We arrange tests to avoid the slice [:0], which doesn’t work, while [0:] is fine.
        msk_row_slice = np.s_[:row_dsp_int] if row_dsp_int < 0 else np.s_[row_dsp_int:]  # type: ignore[index]
        msk_col_slice = np.s_[:col_dsp_int] if col_dsp_int < 0 else np.s_[col_dsp_int:]  # type: ignore[index]

        criteria_row_slice = np.s_[-row_dsp_int:] if row_dsp_int <= 0 else np.s_[:-row_dsp_int]  # type: ignore[index]
        criteria_col_slice = np.s_[-col_dsp_int:] if col_dsp_int <= 0 else np.s_[:-col_dsp_int]  # type: ignore[index]

        mask_img_shape[:] = 0
        mask_img_shape[criteria_row_slice, criteria_col_slice] |= mask_criteria_right[msk_row_slice, msk_col_slice]

        # We subtract the first row and col coordinates to keep the correct indexes when processing a ROI.
        criteria_dataarray.loc[
            {
                "disp_row": row_disp,
                "disp_col": col_disp,
            }
        ] |= mask_img_shape[
            criteria_dataarray.coords["row"].data[:, None] - mask_criteria_right.coords["row"].data[0],
            criteria_dataarray.coords["col"].data[None, :] - mask_criteria_right.coords["col"].data[0],
        ]


def apply_nodata_right_criteria_mask(
    criteria_dataarray: xr.DataArray, mask_criteria_right: xr.DataArray, spline_order_filter: int
) -> None:  # pylint: disable=too-many-branches
    """
    This method apply mask_criteria_right array on criteria_dataarray according
    to row and column disparities.
    The P2D_RIGHT_NODATA criterion is raised for sub-pixel disparities if a no-data value from the
    input mask was used to interpolate the corresponding point.

    :param criteria_dataaray: criteria dataarray to update
    :param mask_criteria_right: mask to apply to criteria dataarray
    :param spline_order_filter: order of the scipy filter
    """

    # We use a temporary mask of the same size as the image to correctly handle cases where the step is different from 1
    mask_img_shape = np.zeros_like(mask_criteria_right, dtype=np.uint8)

    # Create a temporary mask to store the mask dilation using the scipy filter for these disparities
    # Dilation is applied only to floating-point disparities (when subpix > 1)
    # Margins used for filters based on the order (here, the spline_order_filter parameter):
    #     * order = 1 -> Margins(0,0,1,1)
    #     * order = 2 -> Margins(1,1,1,1)
    #     * order = 3 -> Margins(1,1,2,2)
    #     * order = 4 -> Margins(2,2,2,2)
    #     * order = 5 -> Margins(2,2,3,3)
    dilated_mask = np.uint8(Criteria.P2D_RIGHT_NODATA) * binary_dilation(
        mask_criteria_right == np.uint8(Criteria.P2D_RIGHT_NODATA),
        structure=np.ones((spline_order_filter + 1, spline_order_filter + 1)),
        iterations=1,
    )

    def get_mask_slice(disp: np.floating, disp_int: int) -> slice:
        """Returns the consistent slices"""
        if disp >= 0:
            start = disp_int
            # floor(-disp) + floor(disp); this is due to how scipy handles floating-point values
            #  |_ integer values -> 0
            #  |_ floating-point values -> -1
            stop = None if disp.is_integer() else -1
        else:
            # -floor(disp) + floor(disp) = 0 # NOSONAR
            start = None
            stop = disp_int
        return np.s_[start:stop]

    for row_disp, col_disp in itertools.product(
        criteria_dataarray.coords["disp_row"].data, criteria_dataarray.coords["disp_col"].data
    ):
        # np.floor rounds values down to the nearest integer (used in scipy filter)
        # For more information, see the following page:
        # https://docs.scipy.org/doc/scipy/tutorial/ndimage.html#interpolation-boundary-handling
        row_dsp_int, col_dsp_int = int(np.floor(row_disp)), int(np.floor(col_disp))

        # reset temporary mask
        mask_img_shape[:] = 0

        if not row_disp.is_integer() or not col_disp.is_integer():
            mask_criteria = dilated_mask
        else:
            mask_criteria = mask_criteria_right.data

        mask_row_slice = get_mask_slice(row_disp, row_dsp_int)
        mask_col_slice = get_mask_slice(col_disp, col_dsp_int)

        # We arrange tests to avoid the slice [:0], which doesn’t work, while [0:] is fine.
        inv_row_dsp_int, inv_col_dsp_int = int(np.floor(-row_disp)), int(np.floor(-col_disp))
        criteria_row_slice = np.s_[-row_dsp_int:] if row_disp <= 0 else np.s_[:inv_row_dsp_int]  # type: ignore[index]
        criteria_col_slice = np.s_[-col_dsp_int:] if col_disp <= 0 else np.s_[:inv_col_dsp_int]  # type: ignore[index]

        mask_img_shape[criteria_row_slice, criteria_col_slice] |= mask_criteria[mask_row_slice, mask_col_slice]

        # We subtract the first row and col coordinates to keep the correct indexes when processing a ROI.
        criteria_dataarray.loc[
            {
                "disp_row": row_disp,
                "disp_col": col_disp,
            }
        ] |= mask_img_shape[
            criteria_dataarray.coords["row"].data[:, None] - mask_criteria_right.coords["row"].data[0],
            criteria_dataarray.coords["col"].data[None, :] - mask_criteria_right.coords["col"].data[0],
        ]


def apply_peak_on_edge(
    validity_map: xr.DataArray,
    left_image: xr.Dataset,
    cv_coords: tuple[NDArray, NDArray],
    row_map: NDArray,
    col_map: NDArray,
) -> None:
    """
    This method raises P2D_PEAK_ON_EDGE criteria for points (row, col)
    for which the best matching cost is found for the edge of the disparity range.
    This criteria is applied on point (row, col), for each disparity value.

    :param validity_map: 3D validity map
    :param left_image: left image
    :param cv_coords: cost volumes row and column coordinates
    :param row_map: row disparity map
    :param col_map: col disparity map
    """

    # Get disparity grids according to cost volumes coordinates
    d_min_row_grid, d_max_row_grid, d_min_col_grid, d_max_col_grid = get_disparity_grids(left_image, cv_coords)

    # Get P2D_PEAK_ON_EDGE and validity_mask bands
    peak_on_edge_band = validity_map.loc[{"criteria": Criteria.P2D_PEAK_ON_EDGE.name}].data
    validity_mask_band = validity_map.loc[{"criteria": "validity_mask"}].data

    # Condition on peak on edges
    edge_condition = np.logical_or.reduce(
        [row_map == d_min_row_grid, row_map == d_max_row_grid, col_map == d_min_col_grid, col_map == d_max_col_grid]
    )

    # Fill P2D_PEAK_ON_EDGE band in validity dataset
    peak_on_edge_band[edge_condition] = 1

    # Update global validity mask according to P2D_PEAK_ON_EDGE band.
    global_validity_mask = (validity_mask_band == 0) & (peak_on_edge_band == 1)
    validity_mask_band[global_validity_mask] = 1


def allocate_validity_dataset(criteria_dataarray: xr.DataArray) -> xr.Dataset:
    """
    Allocate the validity dataset which contains an additional 'criteria' dimension.

    :param criteria_dataarray: criteria_dataarray used to create validity mask
    """

    # Get criteria names to stock them in the 'criteria' coordinate in the allocated xr.Dataset
    # We use all the criteria, to which we add a band for valid points and a second band for partially valid points
    criteria_names = ["validity_mask"] + ["partial_validity_mask"] + list(Criteria.__members__.keys())[1:]

    coords = {
        "row": criteria_dataarray.coords.get("row"),
        "col": criteria_dataarray.coords.get("col"),
        "criteria": criteria_names,
    }

    dims = ("row", "col", "criteria")
    shape = (len(coords["row"]), len(coords["col"]), len(coords["criteria"]))

    # Initialize validity dataset data with zeros
    empty_data = np.full(shape, 0, dtype=np.uint8)

    dataset = xr.Dataset({"validity": xr.DataArray(empty_data, dims=dims, coords=coords)})

    return dataset


def get_validity_dataset(criteria_dataarray: xr.DataArray, row_disparity: list, col_disparity: list) -> xr.Dataset:
    """
    Fill the validity dataset which contains two additional 'criteria' dimensions.

    :param criteria_dataarray: criteria_dataarray used to create validity mask
    :param row_disparity: user row disparity
    :param col_disparity: user col disparity
    :return: validity Dataset
    """

    validity_dataset = allocate_validity_dataset(criteria_dataarray)

    # subset with real user disparities
    subset = criteria_dataarray.sel(
        disp_row=slice(row_disparity[0], row_disparity[1]),  # Interval row_disparity for disp_row
        disp_col=slice(col_disparity[0], col_disparity[1]),  # Interval col_disparity for disp_col
    )

    # Fill overall mask (valid and partial)
    validity_dataset["validity"].loc[{"criteria": "validity_mask"}] = get_validity_mask_band(subset)
    validity_dataset["validity"].loc[{"criteria": "partial_validity_mask"}] = get_partial_validity_mask_band(subset)

    # invalidating criteria do not depend on disparities,
    # so we can use criteria_datarray at the first couple of disparities
    # to identify the points where the criteria is raised.
    for criterion in DISPARITY_INDEPENDENT_CRITERIA:
        validity_dataset["validity"].loc[{"criteria": criterion.name}] = criterion.is_in(
            criteria_dataarray[:, :, 0, 0].data
        )

    disparity_axis_num = criteria_dataarray.get_axis_num(("disp_row", "disp_col"))
    for criterion in DISPARITY_DEPENDENT_CRITERIA:
        np.logical_or.reduce(
            criterion.is_in(criteria_dataarray.data),
            axis=disparity_axis_num,
            out=validity_dataset["validity"].loc[{"criteria": criterion.name}].data,
        )

    return validity_dataset


def get_validity_mask_band(criteria_dataarray: xr.DataArray) -> NDArray:
    """
    This method fills the validity mask band according to the criteria dataarray given as a parameter.

    This validity mask shows which points of the image are valid and which are not:

    A point marked 1 is invalid, wether not calculated (masked or out of range)
    or not all disparity range resquested by the user have been computed
    A point marked 0 is valid, all the disparity range requested

    A point marked 0 is valid, all the disparity range requested
    by the user have been computed.
    All other points are marked 1.

    :param criteria_dataarray: 4D DataArray containing the criteria
    :return: validity mask band
    """

    disparity_axis_num = criteria_dataarray.get_axis_num(("disp_row", "disp_col"))

    # Fill invalids (at least one criteria in disparities):
    validity_mask = np.logical_or.reduce(criteria_dataarray.data, axis=disparity_axis_num).astype(np.uint8)

    return validity_mask


def get_partial_validity_mask_band(criteria_dataarray: xr.DataArray) -> NDArray:
    """
    This method fills the partial validity mask band according to the criteria dataarray given as a parameter.

    This partial validity mask shows which points of the image are partial or completely invalid :

    A point marked 1 is invalid, all disparity range requested
    by the user can not be computed

    :param criteria_dataarray: 4D DataArray containing the criteria
    :return: partial validity mask band
    """

    disparity_axis_num = criteria_dataarray.get_axis_num(("disp_row", "disp_col"))

    # Fill invalids (all disparities has a criteria):
    partial_validity_mask = np.logical_and.reduce(criteria_dataarray.data, axis=disparity_axis_num).astype(np.uint8)

    return partial_validity_mask
