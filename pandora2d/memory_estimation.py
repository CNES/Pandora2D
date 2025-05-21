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
This module contains methods associated to the pandora2d memory estimation
"""

from typing import Tuple, Dict, List
import numpy as np
from pandora.img_tools import rasterio_open
from pandora.margins import Margins
from pandora2d.img_tools import get_margins_values, get_initial_disparity, get_extrema_disparity

BYTE_TO_MB = 1024 * 1024

# Data variables in image datasets
IMG_DATA_VAR = ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max"]
# Data variables in image datasets when using input mask
IMG_DATA_VAR_MASK = ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max", "msk"]

data_vars_type_size = {
    "im": np.float32().nbytes,
    "row_disparity_min": np.float32().nbytes,
    "row_disparity_max": np.float32().nbytes,
    "col_disparity_min": np.float32().nbytes,
    "col_disparity_max": np.float32().nbytes,
    "msk": np.int16().nbytes,
}


def get_img_size(img_path: str, roi: Dict = None) -> Tuple[int, int]:
    """
    Get width and height from an image path.
    If a ROI is given, its width and height are returned without takin margins into account.

    :param img_path: img path
    :type img_path: str
    :return:  width and height of the image
    :rtype: Tuple[int,int]
    """

    # Get image width and height
    img_read = rasterio_open(img_path)
    width, height = img_read.width, img_read.height

    # Get ROI width and height
    if roi is not None:
        col_off = max(roi["col"]["first"], 0)  # if overlapping on left side
        row_off = max(roi["row"]["first"], 0)  # if overlapping on up side
        roi_width = roi["col"]["last"] - col_off + 1
        roi_height = roi["row"]["last"] - row_off + 1

        # check roi outside
        if col_off > width or row_off > height or (col_off + roi_width) < 0 or (row_off + roi_height) < 0:
            raise ValueError("Roi specified is outside the image")

        # overlap roi and image
        # right side
        if (col_off + roi_width) > width:
            roi_width = width - col_off
        # down side
        if (row_off + roi_height) > height:
            roi_height = height - row_off

        width = roi_width
        height = roi_height

    return height, width


def get_nb_disp(row_disparity: Dict, col_disparity: Dict) -> Tuple[int, int]:
    """
    Get number of row and col disparities

    :param row_disparity: init and range for disparities in rows.
    :type row_disparity: Dict
    :param col_disparity: init and range for disparities in columns.
    :type col_disparity: Dict
    :return:  number of row disparities and number of column disparities
    :rtype: Tuple[int,int]
    """

    # Get initial disparity values
    disparity_row_init = get_initial_disparity(row_disparity)
    disparity_col_init = get_initial_disparity(col_disparity)

    # Get minimum and maximum disparities
    disp_min_row, disp_max_row = get_extrema_disparity(disparity_row_init, row_disparity["range"])
    disp_min_col, disp_max_col = get_extrema_disparity(disparity_col_init, col_disparity["range"])

    # Get number of disparities
    nb_disp_row = disp_max_row - disp_min_row + 1
    nb_disp_col = disp_max_col - disp_min_col + 1

    return nb_disp_row, nb_disp_col


def get_roi_margins(row_disparity, col_disparity, global_margins: Margins) -> Margins:
    """
    Get ROI margins according to row and col disparities and global margins calculated in the check conf step.

    :param row_disparity: init and range for disparities in rows.
    :type row_disparity: Dict
    :param col_disparity: init and range for disparities in columns.
    :type col_disparity: Dict
    :param global_margins: global image margins computed in the check conf
    :type global_margins: Margins
    :return: ROI margins updated according to disparity values
    :rtype: Margins
    """

    # Get initial disparity values
    disparity_row_init = get_initial_disparity(row_disparity)
    disparity_col_init = get_initial_disparity(col_disparity)

    # Get margins for columns
    left, right = get_margins_values(
        disparity_col_init, col_disparity["range"], [global_margins.left, global_margins.right]
    )

    # Get margins for rows
    up, down = get_margins_values(disparity_row_init, row_disparity["range"], [global_margins.up, global_margins.down])

    return Margins(left, up, right, down)


def img_dataset_size(height: int, width: int, sum_nb_bytes: int) -> float:
    """
    Returns image dataset size (MB) according to width, height and sum of the number of bytes corresponding
    to the different data types contained in the image dataset.

    :param height: image or ROI number of rows
    :type height: int
    :param width: image or ROI number of columns
    :type width: int
    :param sum_nb_bytes: sum of the number of bytes.
    :type sum_nb_bytes: int
    :return: size of image dataset in MB
    :rtype: float
    """

    return (height * width * (sum_nb_bytes)) / BYTE_TO_MB


def input_size(height: int, width: int, data_vars: List[str]) -> float:
    """
    Returns input configuration size (MB) according to image width, height
    and data variables contained in the image dataset.

    :param height: image or ROI number of rows
    :type height: int
    :param width: image or ROI number of columns
    :type width: int
    :param data_vars: data variables contained in the image dataset.
    :type data_vars: List of str
    :return: size of image dataset in MB
    :rtype: float
    """

    sum_nb_bytes = 0

    # Compute input configuration size according to each data variable contained in the image dataset
    for data in data_vars:
        sum_nb_bytes += data_vars_type_size[data]

    return img_dataset_size(height, width, sum_nb_bytes)
