#  Copyright (c) 2025. CS GROUP France
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Functions to report information."""

from typing import Dict

import xarray as xr

from .statistics import compute_statistics


def report_disparities(data: xr.Dataset) -> Dict:
    """
    Report statistics on disparities.
    :param data: disparities to report statistics from.
    :type data: xr.Dataset
    :return: dictionary with reported data
    :rtype: Dict
    """
    row_stats = compute_statistics(data["row_map"].data, data.attrs["invalid_disp"])
    col_stats = compute_statistics(data["col_map"].data, data.attrs["invalid_disp"])
    return {"row": row_stats.to_dict(), "col": col_stats.to_dict()}
