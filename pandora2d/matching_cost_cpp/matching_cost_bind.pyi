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

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from ..common_cpp.common_bind import CostVolumeSize

def compute_cost_volumes_cpp(
    left: NDArray[np.float64],
    right: List[NDArray[np.float64]],
    cv_values: NDArray[np.float64],
    cv_size: CostVolumeSize,
    disp_range_row: NDArray[np.float64],
    disp_range_col: NDArray[np.float64],
    offset_cv_img_row: int,
    offset_cv_img_col: int,
    window_size: int,
    step: NDArray[np.integer],
    no_data: float,
):
    """
    Computes the cost values

    :param left: left image
    :type left: NDArray[np.float64]
    :param right: list of right images
    :type right: List[NDArray[np.float64]]
    :param cv_values:  cost volumes initialized values
    :type cv_values: NDArray[np.float64]
    :param cv_size: cost_volume size [nb_row, nb_col, nb_disp_row, nb_disp_col]
    :type cv_size: CostVolumeSize
    :param disp_range_row:  cost volumes row disparity range
    :type disp_range_row: NDArray[np.float64]
    :param disp_range_col:  cost volumes col disparity range
    :type disp_range_col: NDArray[np.float64]
    :param offset_cv_img_row: row offset between first index of cv and image (ROI case)
    :type offset_cv_img_row: int
    :param offset_cv_img_col: col offset between first index of cv and image (ROI case)
    :type offset_cv_img_col: int
    :param window_size: size of the correlation window
    :type window_size: int
    :param step: [step_row, step_col]
    :type step: NDArray[np.integer]
    :param no_data: no data value in img
    :type no_data: float
    """
