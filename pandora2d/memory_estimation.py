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

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import DTypeLike

from pandora.img_tools import rasterio_open

from pandora2d.constants import Criteria
from pandora2d.img_tools import get_extrema_disparity, get_initial_disparity, get_margins_values
from pandora2d.margins import Margins, NullMargins

BYTE_TO_MB = 1024 * 1024

RELATIVE_ESTIMATION_MARGIN = 0.4

# Data variables in image datasets
IMG_DATA_VAR = ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max"]
# Data variables in image datasets when using input mask
IMG_DATA_VAR_MASK = ["im", "row_disparity_min", "row_disparity_max", "col_disparity_min", "col_disparity_max", "msk"]
# Data variables in float32 cost volumes
CV_FLOAT_DATA_VAR = ["cost_volumes_float", "criteria"]
# Data variables in float64/double cost volumes
CV_DOUBLE_DATA_VAR = ["cost_volumes_double", "criteria"]

DATA_VARS_TYPE_SIZE = {
    "im": np.float32().nbytes,
    "row_disparity_min": np.float32().nbytes,
    "row_disparity_max": np.float32().nbytes,
    "col_disparity_min": np.float32().nbytes,
    "col_disparity_max": np.float32().nbytes,
    "msk": np.int16().nbytes,
    "cost_volumes_float": np.float32().nbytes,
    "cost_volumes_double": np.float64().nbytes,
    "criteria": np.uint8().nbytes,
}


def estimate_total_consumption(config: Dict, margin_disp: Margins = NullMargins()) -> float:
    """
    Estimate the total memory consumption of all objects that will be allocated.
    :param config: configuration with ROI margins if necessary.
    :type config: Dict
    :param margin_disp: Disparity margins.
    :type margin_disp: Margins
    :return: Estimated memory consumption in Mbytes.
    :rtype: float
    """
    height, width = get_img_size(config["input"]["left"]["img"])
    if global_margins := config.get("ROI", {}).get("margins"):
        roi_margins = get_roi_margins(
            config["input"]["row_disparity"], config["input"]["col_disparity"], global_margins
        )
    else:
        roi_margins = NullMargins()

    height += roi_margins.up + roi_margins.down
    width += roi_margins.left + roi_margins.right

    cost_volume_dtype = np.dtype(config["pipeline"]["matching_cost"]["float_precision"])
    cost_volume_datavars = CV_FLOAT_DATA_VAR if cost_volume_dtype == np.float32 else CV_DOUBLE_DATA_VAR

    # A copy of pandora cost volume is done in calculations so it counts twice:
    number_of_pandora_cost_volumes = 2

    return (
        estimate_input_size(height, width, IMG_DATA_VAR)
        + estimate_cost_volumes_size(config, height, width, margin_disp, cost_volume_datavars)
        + (
            number_of_pandora_cost_volumes * estimate_pandora_cost_volume_size(config, height, width, margin_disp)
            if config["pipeline"]["matching_cost"]["matching_cost_method"] != "mutual_information"
            else 0
        )
        + (
            estimate_shifted_right_images_size(height, width, subpix)  # pylint: disable=used-before-assignment
            if (subpix := config["pipeline"]["matching_cost"]["subpix"]) > 1
            else 0
        )
        + estimate_dataset_disp_map_size(config, height, width, cost_volume_dtype)
    )


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


def get_nb_disp(disparity: Dict, before_margins: int = 0, after_margins: int = 0, subpix: int = 1) -> int:
    """
    Get number of disparities.

    :param disparity: init and range for disparities.
    :type disparity: Dict
    :param before_margins: Margins before the minimum disparity.
    :type before_margins: int
    :param after_margins: Margins after the maximum disparity.
    :type after_margins: int
    :param subpix: subpix
    :type subpix: int
    :return:  number of disparities
    :rtype: int
    """

    # Get initial disparity values
    initial_disparity = get_initial_disparity(disparity)

    # Get minimum and maximum disparities
    min_disparity, max_disparity = get_extrema_disparity(initial_disparity, disparity["range"])

    # Get number of disparities
    return (max_disparity - min_disparity + before_margins + after_margins) * subpix + 1


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


def img_dataset_size(height: int, width: int, nb_bytes: int) -> float:
    """
    Return image dataset size (MB) according to width, height and sum of the number of bytes corresponding
    to the different data types contained in the image dataset.

    :param height: image or ROI number of rows
    :type height: int
    :param width: image or ROI number of columns
    :type width: int
    :param nb_bytes: sum of the number of bytes.
    :type nb_bytes: int
    :return: size of image dataset in MB
    :rtype: float
    """

    return (height * width * (nb_bytes)) / BYTE_TO_MB


def estimate_input_size(height: int, width: int, data_vars: List[str]) -> float:
    """
    Estimate input configuration size (MB) according to image width, height
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

    # Compute input configuration size according to each data variable contained in the image dataset
    nb_bytes = sum(DATA_VARS_TYPE_SIZE[data_var] for data_var in data_vars)

    return img_dataset_size(height, width, nb_bytes)


def estimate_cost_volumes_size(
    user_cfg: Dict, height: int, width: int, margins_disp: Margins, data_vars: List[str]
) -> float:
    """
    Estimate 4D cost volumes size (MB) according to image width, height,
    number of disparities, subpix, step and data variables contained in the cost volumes dataset.

    :param user_cfg: user configuration
    :type user_cfg: Dict
    :param height: image or ROI number of rows
    :type height: int
    :param width: image or ROI number of columns
    :type width: int
    :param margins_disp: disparity margins computed in the check conf
    :type margins_disp: Margins
    :param data_vars: data variables contained in the cost_volumes dataset.
    :type data_vars: List of str
    :return: size of image dataset in MB
    :rtype: float
    """

    # Get cost volumes parameters used in size estimation
    subpix = user_cfg["pipeline"]["matching_cost"]["subpix"]
    step = user_cfg["pipeline"]["matching_cost"]["step"]

    nb_disp_row = get_nb_disp(user_cfg["input"]["row_disparity"], margins_disp.up, margins_disp.down, subpix)
    nb_disp_col = get_nb_disp(user_cfg["input"]["col_disparity"], margins_disp.left, margins_disp.right, subpix)

    # Get cost volumes shape
    cv_shape = np.ceil(height / step[0]) * np.ceil(width / step[1]) * nb_disp_row * nb_disp_col

    nb_bytes = sum(DATA_VARS_TYPE_SIZE[data_var] for data_var in data_vars)

    cv_size = nb_bytes * cv_shape / BYTE_TO_MB

    return cv_size


def estimate_shifted_right_images_size(height: int, width: int, subpix: int) -> float:
    """
    Estimate the size in MB of the list of shifted right images (excluding the original right image itself).

    :param height: height of image
    :type height: int
    :param width: width of image
    :type width: int
    :param subpix: subpixel
    :type subpix:
    :return: estimated size in MB
    :rtype: float
    """
    one_image_size = img_dataset_size(height, width, DATA_VARS_TYPE_SIZE["im"])
    # When subpix is 1, no new image is created; instead, a reference to the original right image is used.
    # As a result, even though the list of shifted right images contains `subpix * subpix` images,
    # we need to take into account one less image in the memory estimation:
    number_of_images = subpix * subpix - 1
    return one_image_size * number_of_images


def estimate_pandora_cost_volume_size(config: Dict, height: int, width: int, margins: Margins) -> float:
    """
    Estimate the size in MB of the cost volume according to image width, height, and refinement margins.

    :param config: user configuration.
    :type config: Dict
    :param height: image or ROI number of rows
    :type height: int
    :param width: image or ROI number of columns
    :type width: int
    :param margins: Refinement margins.
    :type margins: Margins
    :return: estimated size in MB.
    :rtype: float
    """
    subpix = config["pipeline"]["matching_cost"]["subpix"]
    disparity_size = get_nb_disp(config["input"]["col_disparity"], margins.left, margins.right, subpix)

    image_size = height * width

    return DATA_VARS_TYPE_SIZE["cost_volumes_float"] * image_size * disparity_size / BYTE_TO_MB


def estimate_dataset_disp_map_size(config: Dict, height: int, width: int, dtype: DTypeLike) -> float:
    """
    Estimate the size in MB of the disparity map dataset.

    :param config: user configuration.
    :type config: Dict
    :param height: image or ROI number of rows.
    :type height: int
    :param width: image or ROI number of columns.
    :type width: int
    :param dtype: dtype of the disparity map (should be same as cost volumes dataset).
    :type dtype: np.typing.DTypeLike
    :return: estimated size in MB.
    :rtype: float
    """
    step = config["pipeline"]["matching_cost"]["step"]
    image_size = np.ceil(height / step[0]) * np.ceil(width / step[1])
    number_of_dtyped_datavars = 3  # row_map, col_map, correlation_score
    # The number of criteria is incremented by one in order to take the validity_mask band into account:
    number_of_validity_bands = len(Criteria.__members__) + 1
    data_vars_size = (
        number_of_dtyped_datavars * np.dtype(dtype).itemsize
        + number_of_validity_bands * DATA_VARS_TYPE_SIZE["criteria"]
    )
    return image_size * data_vars_size / BYTE_TO_MB
