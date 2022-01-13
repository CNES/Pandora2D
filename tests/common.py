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
        "img_left": "./tests/data/left.png",
        "img_right": "./tests/data/right.png",
        "nodata_left": "NaN",
        "disp_min_col": -2,
        "disp_max_col": 2,
        "disp_min_row": -2,
        "disp_max_row": 2,
    }
}


false_input_path_image = {
    "input": {
        "img_left": "./tests/data/lt.png",
        "img_right": "./tests/data/right.png",
        "nodata_left": "NaN",
        "disp_min_col": -2,
        "disp_max_col": 2,
        "disp_min_row": -2,
        "disp_max_row": 2,
    }
}

false_input_disp = {
    "input": {
        "img_left": "./tests/data/left.png",
        "img_right": "./tests/data/right.png",
        "disp_min_col": 7,
        "disp_max_col": 2,
        "disp_min_row": -2,
        "disp_max_row": 2,
    }
}

correct_pipeline = {
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
    "refinement": {"refinement_method": "interpolation"},
}

false_pipeline_mc = {
    "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
    "refinement": {"refinement_method": "interpolation"},
}

false_pipeline_disp = {
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
    "refinement": {"refinement_method": "interpolation"},
}

correct_pipeline_dict = {
    "pipeline": {
        "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
        "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        "refinement": {"refinement_method": "interpolation"},
        }
}

false_pipeline_mc_dict = {
    "pipeline": {
        "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
        "refinement": {"refinement_method": "interpolation"},
        }
}

false_pipeline_disp_dict = {
    "pipeline": {
        "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
        "refinement": {"refinement_method": "interpolation"},
        }
}