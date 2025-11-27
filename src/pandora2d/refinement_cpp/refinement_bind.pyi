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

# pylint: skip-file
import numpy as np
from numpy.typing import NDArray

from ..interpolation_filter_cpp.interpolation_filter_bind import AbstractFilter

def compute_dichotomy_float(
    cost_volume: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.floating],
    criteria_map: NDArray[np.floating],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    method_matching_cost: str,
) -> None:
    """
    Dichotomy calculation with float data

    :param cost_volume: cost volume data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param criteria_map: criteria map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param method_matching_cost: max or min
    """

def compute_dichotomy_double(
    cost_volume: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    invalid_map: NDArray[np.floating],
    criteria_map: NDArray[np.floating],
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    method_matching_cost: str,
) -> None:
    """
    Dichotomy calculation with double data

    :param cost_volume: cost volume data
    :param disparity_map_col: column disparity map data
    :param disparity_map_row: row disparity map data
    :param score_map: score map data
    :param invalid_map: invalid map data
    :param criteria_map: criteria map data
    :param subpixel: sub-sampling of cost_volume
    :param nb_iterations: number of iterations of the dichotomy
    :param filter: interpolation filter
    :param method_matching_cost: max or min
    """
