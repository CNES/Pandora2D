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

# pylint: skip-file
import numpy as np
from numpy.typing import NDArray
from typing import overload
from ..interpolation_filter_cpp.interpolation_filter_bind import AbstractFilter

class Cost_volume_size:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, cv_size: NDArray[np.floating]) -> None: ...
    @overload
    def __init__(self, _r: int, _c: int, _dr: int, _dc: int) -> None: ...
    def size(self) -> int:
        """
        Returns the cost_volume size nb_row * nb_col * nb_disp_row * nb_disp_col.
        """

    def nb_disps(self) -> int:
        """
        Returns the disparity number : nb_disp_row * nb_disp_col.
        """

def compute_dichotomy(
    cost_volume: NDArray[np.floating],
    disparity_map_col: NDArray[np.floating],
    disparity_map_row: NDArray[np.floating],
    score_map: NDArray[np.floating],
    criteria_map: NDArray[np.floating],
    cv_size: Cost_volume_size,
    subpixel: int,
    nb_iterations: int,
    filter: AbstractFilter,
    method_matching_cost: str,
) -> None:
    """
    Dichotomy calculation

    :param cost_volume: cost volume data
    :type cost_volume: NDArray[np.floating]
    :param disparity_map_col: column disparity map data
    :type disparity_map_col: NDArray[np.floating]
    :param disparity_map_row: row disparity map data
    :type disparity_map_row: NDArray[np.floating]
    :param score_map: score map data
    :type score_map: NDArray[np.floating]
    :param criteria_map: criteria map data
    :type criteria_map: NDArray[np.floating]
    :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
    :type cv_size: Cost_volume_size
    :param subpixel: sub-sampling of cost_volume
    :type subpixel: int
    :param nb_iterations: number of iterations of the dichotomy
    :type nb_iterations: int
    :param filter: interpolation filter
    :type filter: abstractfilter::AbstractFilter
    :param method_matching_cost: max or min
    :type method_matching_cost: str
    """
