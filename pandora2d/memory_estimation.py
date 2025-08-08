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

import math
from typing import Dict, List, Tuple, TypedDict

import numpy as np
from numpy.typing import DTypeLike
from pandora.img_tools import rasterio_open

from pandora2d.constants import Criteria
from pandora2d.img_tools import get_extrema_disparity, get_initial_disparity, get_margins_values
from pandora2d.margins import Margins, NullMargins
from pandora2d.matching_cost import MatchingCostRegistry

BYTE_TO_MB = 1024 * 1024

RELATIVE_ESTIMATION_MARGIN = 0.25

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


def estimate_total_consumption(config: Dict, height: int, width: int, margin_disp: Margins = NullMargins()) -> float:
    """
    Estimate the total memory consumption of all objects that will be allocated.
    :param config: configuration with ROI margins if necessary.
    :type config: Dict
    :param height: Image height including any ROI adjustments.
    :type height: int
    :param width: Image width including any ROI adjustments.
    :type width: int
    :param margin_disp: Disparity margins.
    :type margin_disp: Margins
    :return: Memory consumption estimate in megabytes.
    :rtype: float
    """
    matching_cost_config = config["pipeline"]["matching_cost"]
    cost_volume_dtype = np.dtype(matching_cost_config["float_precision"])
    cost_volume_datavars = CV_FLOAT_DATA_VAR if cost_volume_dtype == np.float32 else CV_DOUBLE_DATA_VAR

    # Left and Right images
    number_of_images = 2
    number_of_pandora_cost_volumes = 1

    result = (
        number_of_images * estimate_input_size(height, width, IMG_DATA_VAR)
        + estimate_cost_volumes_size(config, height, width, margin_disp, cost_volume_datavars)
        + estimate_dataset_disp_map_size(height, width, matching_cost_config["step"], cost_volume_dtype)
    )

    if matching_cost_config["matching_cost_method"] not in MatchingCostRegistry.registered:
        result += number_of_pandora_cost_volumes * estimate_pandora_cost_volume_size(config, height, width, margin_disp)

    subpix = matching_cost_config["subpix"]
    if subpix > 1:
        result += estimate_shifted_right_images_size(height, width, subpix)

    return result


def compute_effective_image_size(config: Dict, image_margins: Margins) -> Tuple[int, int]:
    """
    Compute the effective image size (height, width), including ROI and global margins.

    :param config: Configuration dictionary containing the image path and optional ROI information.
    :type config: Dict
    :param image_margins: Margins to apply around the ROI to ensure the full region is processed.
                          Used only when a ROI is defined. Defaults to None.
    :type image_margins: Margins or None
    :return: Image dimensions as (height, width) including margins.
    :rtype: Tuple[int, int]
    """
    height, width = get_img_size(config["input"]["left"]["img"], config.get("ROI"))
    if "ROI" in config:
        roi_margins = get_roi_margins(
            config["input"]["row_disparity"],
            config["input"]["col_disparity"],
            image_margins,
        )
    else:
        roi_margins = NullMargins()
    height += roi_margins.up + roi_margins.down
    width += roi_margins.left + roi_margins.right
    return height, width


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
    cv_shape = math.ceil(height / step[0]) * math.ceil(width / step[1]) * nb_disp_row * nb_disp_col

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
    step = config["pipeline"]["matching_cost"]["step"]
    disparity_size = get_nb_disp(config["input"]["col_disparity"], margins.left, margins.right, subpix)

    image_size = height * math.ceil(width / step[1])

    return DATA_VARS_TYPE_SIZE["cost_volumes_float"] * image_size * disparity_size / BYTE_TO_MB


def estimate_dataset_disp_map_size(height: int, width: int, step: List, dtype: DTypeLike) -> float:
    """
    Estimate the size in MB of the disparity map dataset.

    :param height: image or ROI number of rows.
    :type height: int
    :param width: image or ROI number of columns.
    :type width: int
    :param step: step.
    :type step: List
    :param dtype: dtype of the disparity map (should be same as cost volumes dataset).
    :type dtype: np.typing.DTypeLike
    :return: estimated size in MB.
    :rtype: float
    """
    image_size = math.ceil(height / step[0]) * math.ceil(width / step[1])
    number_of_dtyped_datavars = 3  # row_map, col_map, correlation_score
    # The number of criteria is incremented by one in order to take the validity_mask band into account:
    number_of_validity_bands = len(Criteria.__members__) + 1
    data_vars_size = (
        number_of_dtyped_datavars * np.dtype(dtype).itemsize
        + number_of_validity_bands * DATA_VARS_TYPE_SIZE["criteria"]
    )
    return image_size * data_vars_size / BYTE_TO_MB


class RoiRange(TypedDict):
    """
    Represents the range of rows or columns in a region of interest (ROI).

    :param first: Index of the first row or column.
    :type first: int
    :param last: Index of the last row or column (inclusive).
    :type last: int
    """

    first: int
    last: int


class Roi(TypedDict):
    """
    Represents a 2D region of interest, defined by row and column bounds.

    :param row: Row range of the ROI.
    :type row: RoiRange
    :param col: Column range of the ROI.
    :type col: RoiRange
    """

    row: RoiRange
    col: RoiRange


def segment_image_by_rows(config: Dict, disp_margins: Margins, image_margins: Margins) -> List[Roi]:
    """
    Split an image into multiple horizontal ROI segments that fit within memory constraints.

    This function estimates the memory required to process the full image with the provided
    disparity margins. If the memory requirement exceeds the configured `memory_per_work`,
    the image is split into horizontal segments whose individual memory usage remains within
    the allowed limit.

    :param config: Configuration dictionary containing keys such as 'segment_mode' and 'pipeline'.
    :type config: Dict

    :param disp_margins: Margins applied during disparity computation.
                         Defaults to NullMargins.
    :type disp_margins: Margins

    :param image_margins: Margins applied to image.
    :type image_margins: Margins

    :return: List of segment dictionaries with row and column bounds.
    :rtype: List[Roi]

    :raises ValueError: If the minimum memory required for processing a basic segment
                        exceeds the configured `memory_per_work`.
    """

    # Estimate total memory required for full image
    height, width = compute_effective_image_size(config, image_margins)
    whole_image_estimation = estimate_total_consumption(config, height, width, disp_margins)
    asked_memory_per_work = config["segment_mode"].get("memory_per_work")
    estimation_margin_factor = 1 - RELATIVE_ESTIMATION_MARGIN
    memory_per_work = int(estimation_margin_factor * asked_memory_per_work)

    # Bypass segmentation if disabled or memory fits full image
    if whole_image_estimation <= memory_per_work:
        return []

    cost_volume_dtype = np.dtype(config["pipeline"]["matching_cost"]["float_precision"])

    # Estimate fixed memory usage for final disparity map
    final_dataset_disp_map_size = estimate_dataset_disp_map_size(
        height, width, config["pipeline"]["matching_cost"]["step"], cost_volume_dtype
    )

    roi_margins = get_roi_margins(
        config["input"]["row_disparity"],
        config["input"]["col_disparity"],
        image_margins,
    )
    height_margins = roi_margins.up + roi_margins.down
    width_margins = roi_margins.left + roi_margins.right
    min_roi_width = width + width_margins
    min_roi_height = 1 + height_margins
    # Estimate memory needed for smallest possible ROI (1 row)
    min_roi_memory = estimate_total_consumption(config, min_roi_height, min_roi_width, disp_margins)
    min_required_memory = final_dataset_disp_map_size + min_roi_memory

    if min_required_memory > memory_per_work:
        # `memory_per_work` already includes RELATIVE_ESTIMATION_MARGIN.
        # However, in this error message, we want to display the estimated minimum
        # memory requirement *with* the margin applied, for proper comparison with the user-defined value.
        # That’s why we apply the margin to `min_required_memory` before formatting.
        raise ValueError(
            f"estimated minimum `memory_per_work` is "
            f"{math.ceil(min_required_memory / estimation_margin_factor)} MB, "
            f"but got {asked_memory_per_work} MB. Consider increasing it, reducing image size or working on ROI."
        )

    # Compute usable memory per ROI and derive segment size
    max_roi_memory = memory_per_work - final_dataset_disp_map_size
    number_of_pixels = min_roi_width * min_roi_height

    memory_per_pixel = min_roi_memory / number_of_pixels
    max_pixels_per_roi = max_roi_memory // memory_per_pixel
    # Anticipate added vertical margins when opening ROIs:
    # subtract now to prevent oversizing and ensure the estimated number of rows fits memory constraints.
    max_rows_per_roi = max(1, (max_pixels_per_roi // min_roi_width) - height_margins)

    input_roi = config.get("ROI")

    first_row_coordinate = 0 if input_roi is None else input_roi["row"]["first"]
    last_row_coordinate = height if input_roi is None else input_roi["row"]["last"] + 1
    starts = np.arange(first_row_coordinate, last_row_coordinate, max_rows_per_roi)
    ends = (starts + max_rows_per_roi - 1).clip(max=last_row_coordinate - 1)

    col_roi: RoiRange = {
        "first": 0 if input_roi is None else input_roi["col"]["first"],
        "last": width - 1 if input_roi is None else input_roi["col"]["last"],
    }
    # We need to convert to integer because of json_checker expects Python integer and not numpy integer
    return [{"row": {"first": int(s), "last": int(e)}, "col": col_roi} for s, e in zip(starts, ends)]
