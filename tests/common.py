#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
This module contains common functions present in Pandora2D's tests.
"""

correct_input = {
    "input": {
        "left": {
            "img": "./tests/data/left.png",
            "nodata": "NaN",
        },
        "right": {
            "img": "./tests/data/right.png",
        },
        "col_disparity": [-2, 2],
        "row_disparity": [-2, 2],
    }
}


false_input_path_image = {
    "input": {
        "left": {
            "img": "./tests/data/lt.png",
            "nodata": "NaN",
        },
        "right": {
            "img": "./tests/data/right.png",
        },
        "col_disparity": [-2, 2],
        "row_disparity": [-2, 2],
    }
}

false_input_disp = {
    "input": {
        "left": {
            "img": "./tests/data/left.png",
        },
        "right": {
            "img": "./tests/data/right.png",
        },
        "col_disparity": [7, 2],
        "row_disparity": [-2, 2],
    }
}

correct_pipeline = {
    "pipeline": {
        "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
        "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        "refinement": {"refinement_method": "interpolation"},
    }
}

false_pipeline_mc = {
    "pipeline": {
        "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        "refinement": {"refinement_method": "interpolation"},
    }
}

false_pipeline_disp = {
    "pipeline": {
        "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
        "refinement": {"refinement_method": "interpolation"},
    }
}

correct_ROI_sensor = {
    "ROI": {
        "col": {"first": 10, "last": 100},
        "row": {"first": 10, "last": 100},
    }
}

false_ROI_sensor_negative = {
    "ROI": {
        "col": {"first": -10, "last": 100},
        "row": {"first": 10, "last": 100},
    }
}

false_ROI_sensor_first_superior_to_last = {
    "ROI": {
        "col": {"first": 110, "last": 100},
        "row": {"first": 10, "last": 100},
    }
}
