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
This module contains functions associated to the validity mask created in the cost volume step.
"""

import xarray as xr
import numpy as np

from pandora.criteria import binary_dilation_msk


def allocate_criteria_dataset(cv: xr.Dataset, value, data_type) -> xr.Dataset:
    """
    This method creates the criteria_dataset with the same dimensions as cost_volumes (cv).
    Initially, all points are considered valid and have the value XX.

    :param cv: cost_volumes
    :type cv: xarray.Dataset
    :return: criteria_dataset: 4D Dataset containing the criteria
    :rtype: criteria_dataset: xr.Dataset
    """
    return xr.Dataset(
        {
            "criteria": (["row", "col", "disp_col", "disp_row"], np.full(cv.cost_volumes.shape, value, data_type)),
        },
        coords={"row": cv.row.data, "col": cv.col.data, "disp_col": cv.disp_col.data, "disp_row": cv.disp_row.data},
    )
