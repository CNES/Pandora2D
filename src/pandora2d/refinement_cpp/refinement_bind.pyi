# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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

# pylint: skip-file
import numpy as np
from numpy.typing import NDArray

from ..interpolation_filter_cpp.interpolation_filter_bind import AbstractFilter

def compute_dichotomy_cost_surface_float(
    cost_volumes: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.uint8],
    criteria_map: NDArray[np.floating],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    type_method_matching_cost: str,
) -> None:
    """
    Dichotomy calculation on cost surface with float data

    :param cost_volumes: cost volumes data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param criteria_map: criteria map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param type_method_matching_cost: max or min
    """

def compute_dichotomy_cost_surface_double(
    cost_volumes: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.uint8],
    criteria_map: NDArray[np.floating],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    type_method_matching_cost: str,
) -> None:
    """
    Dichotomy calculation on cost surface with double data

    :param cost_volumes: cost volumes data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param criteria_map: criteria map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param type_method_matching_cost: max or min
    """

def compute_dichotomy_image_float(
    nb_cols: int,
    left_image: NDArray[np.floating],
    right_image: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.uint8],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    method_matching_cost: str,
    window_size: int,
    step: list[int],
    offset_cv_img: list[int],
    disp_min: list[float],
    right_valid_mask: NDArray[np.uint8],
) -> None:
    """
    Dichotomy calculation on image with float data

    :param nb_cols: number of columns in the cost volume
    :param left_image: left image data
    :param right_image: right image data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param method_matching_cost: matching cost method
    :param window_size: size of the correlation window
    :param step: step of the matching cost
    :param offset_cv_img: offset of the cost volume in the image
    :param disp_min: minimum disparity values
    :param right_valid_mask: mask of valid pixels in the right image
    """

def compute_dichotomy_image_double(
    nb_cols: int,
    left_image: NDArray[np.floating],
    right_image: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.uint8],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    method_matching_cost: str,
    window_size: int,
    step: list[int],
    offset_cv_img: list[int],
    disp_min: list[float],
    right_valid_mask: NDArray[np.uint8],
) -> None:
    """
    Dichotomy calculation on image with double data

    :param nb_cols: number of columns in the cost volume
    :param left_image: left image data
    :param right_image: right image data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param method_matching_cost: matching cost method
    :param window_size: size of the correlation window
    :param step: step of the matching cost
    :param offset_cv_img: offset of the cost volume in the image
    :param disp_min: minimum disparity values
    :param right_valid_mask: mask of valid pixels in the right image
    """
