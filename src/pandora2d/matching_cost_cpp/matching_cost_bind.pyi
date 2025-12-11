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

from typing import List

import numpy as np
from numpy.typing import NDArray

from ..common_cpp.common_bind import CostVolumeSize

def compute_cost_volumes_cpp_float(
    left: NDArray[np.float32],
    right: List[NDArray[np.float32]],
    cv_values: NDArray[np.floating],
    criteria_values: NDArray[np.uint8],
    cv_size: CostVolumeSize,
    disp_range_row: NDArray[np.float64],
    disp_range_col: NDArray[np.float64],
    offset_cv_img_row: int,
    offset_cv_img_col: int,
    window_size: int,
    step: NDArray[np.integer],
    matching_cost_method: str,
) -> None:
    """
    Computes the cost values in float32

    :param left: left image
    :param right: list of right images
    :param cv_values:  cost volumes initialized values
    :param criteria_values:  criteria values
    :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
    :param disp_range_row:  cost volumes row disparity range
    :param disp_range_col:  cost volumes col disparity range
    :param offset_cv_img_row: row offset between first index of cv and image (ROI case)
    :param offset_cv_img_col: col offset between first index of cv and image (ROI case)
    :param window_size: size of the correlation window
    :param step: [step_row, step_col]
    :param matching_cost_method: correlation method
    """

def compute_cost_volumes_cpp_double(
    left: NDArray[np.float32],
    right: List[NDArray[np.float32]],
    cv_values: NDArray[np.floating],
    criteria_values: NDArray[np.uint8],
    cv_size: CostVolumeSize,
    disp_range_row: NDArray[np.float64],
    disp_range_col: NDArray[np.float64],
    offset_cv_img_row: int,
    offset_cv_img_col: int,
    window_size: int,
    step: NDArray[np.integer],
    matching_cost_method: str,
) -> None:
    """
    Computes the cost values in float64

    :param left: left image
    :param right: list of right images
    :param cv_values:  cost volumes initialized values
    :param criteria_values:  criteria values
    :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
    :param disp_range_row:  cost volumes row disparity range
    :param disp_range_col:  cost volumes col disparity range
    :param offset_cv_img_row: row offset between first index of cv and image (ROI case)
    :param offset_cv_img_col: col offset between first index of cv and image (ROI case)
    :param window_size: size of the correlation window
    :param step: [step_row, step_col]
    :param matching_cost_method: correlation method

    """
