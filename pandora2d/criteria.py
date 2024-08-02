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
This module contains functions associated to the validity mask and criteria dataset created in the cost volume step.
"""

from typing import Union
import xarray as xr
import numpy as np
from numpy.typing import NDArray


from pandora2d.constants import Criteria
from pandora2d.common import (
    set_out_of_col_disparity_range_to_other_value,
    set_out_of_row_disparity_range_to_other_value,
)


def allocate_criteria_dataset(
    cv: xr.Dataset, value: Union[int, float, Criteria] = Criteria.VALID, data_type: Union[np.dtype, None] = None
) -> xr.Dataset:
    """
    This method creates the criteria_dataset with the same dimensions as cost_volumes (cv).
    Initially, all points are considered valid and have the value XX.

    :param cv: cost_volumes
    :type cv: 4D xarray.Dataset
    :param value: value representing the valid criteria, by default Criteria.VALID = 0
    :type value: Union[int, float, Criteria]
    :param data_type: the desired data-type for the criteria_dataset.
    :type data_type: Union[np.dtype, None], by default None
    :return: criteria_dataset: 4D Dataset containing the criteria
    :rtype: criteria_dataset: xr.Dataset
    """
    return xr.Dataset(
        {
            "criteria": (["row", "col", "disp_col", "disp_row"], np.full(cv.cost_volumes.shape, value, data_type)),
        },
        coords={"row": cv.row.data, "col": cv.col.data, "disp_col": cv.disp_col.data, "disp_row": cv.disp_row.data},
    )


def set_unprocessed_disp(
    criteria_dataset: xr.Dataset,
    min_grid_col: NDArray[np.floating],
    max_grid_col: NDArray[np.floating],
    min_grid_row: NDArray[np.floating],
    max_grid_row: NDArray[np.floating],
):
    """
    This method sets PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED to points for disparities that will not be processed,
    based on the disparity grids provided.

    :param criteria_dataset: 4D Dataset containing the criteria
    :type criteria_dataset: xr.Dataset 4D
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
        criteria_dataset, min_grid_col, max_grid_col, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, "criteria"
    )
    # Check row disparity
    set_out_of_row_disparity_range_to_other_value(
        criteria_dataset, min_grid_row, max_grid_row, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED, "criteria"
    )


def mask_border(cost_volumes: xr.Dataset, criteria_dataset: xr.Dataset):
    """
    This method raises PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria on the edges of the criteria_dataset
    for each of the disparities.

    PANDORA2D_MSK_PIXEL_LEFT_BORDER criteria is non-cumulative, so this method will be called last.

    :param cost_volumes: 4D xarray.Dataset
    :type cost_volumes: 4D xarray.Dataset
    :param criteria_dataset: 4D xarray.Dataset with all criteria
    :type criteria_dataset: 4D xarray.Dataset
    """

    offset = cost_volumes.attrs["offset_row_col"]

    if offset > 0:

        # Raise criteria 0 on border of criteria_disp_col according to offset value
        criteria_dataset["criteria"].data[:offset, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataset["criteria"].data[-offset:, :, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataset["criteria"].data[:, :offset, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
        criteria_dataset["criteria"].data[:, -offset:, :, :] = Criteria.PANDORA2D_MSK_PIXEL_LEFT_BORDER
