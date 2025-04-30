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

"""Test report module."""

import numpy as np
import xarray as xr
from pandora2d import reporting


def test_report_disparities():
    """Test result of report_disparities."""
    disparities = xr.Dataset(
        {
            "row_map": (["row", "col"], np.array([[0, 0, -99.0], [20, 20, -99.0]])),
            "col_map": (["row", "col"], np.array([[10, 10, -99.0], [20, 20, -99.0]])),
        },
        coords={
            "row": np.arange(2),
            "col": np.arange(3),
        },
        attrs={"invalid_disp": -99.0},
    )
    result = reporting.report_disparities(disparities)

    assert result == {
        "row": {
            "mean": 10.0,
            "std": 10.0,
            "quantiles": {"p10": 0.0, "p25": 0.0, "p50": 10.0, "p75": 20.0, "p90": 20.0},
            "minimal_valid_pixel_ratio": 2 / 3,
        },
        "col": {
            "mean": 15.0,
            "std": 5.0,
            "quantiles": {"p10": 10.0, "p25": 10.0, "p50": 15.0, "p75": 20.0, "p90": 20.0},
            "minimal_valid_pixel_ratio": 2 / 3,
        },
    }
