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

import numpy as np


correct_input = {
    "input": {
        "img_left": "./data/left.png",
        "img_right": "./data/right.png",
        "no_data": np.nan,
        "disp_min_x": -2,
        "disp_max_x": 2,
        "disp_min_y": -2,
        "disp_max_y": 2,
    }
}

false_input_no_data = {
    "input": {
        "img_left": "./data/left.png",
        "img_right": "./data/right.png",
        "disp_min_x": 5,
        "disp_max_x": 2,
        "disp_min_y": -2,
        "disp_max_y": 2,
    }
}

false_input_path_image = {
    "input": {
        "img_left": "./data/lt.png",
        "img_right": "./data/right.png",
        "no_data": np.nan,
        "disp_min_x": -2,
        "disp_max_x": 2,
        "disp_min_y": -2,
        "disp_max_y": 2,
    }
}

false_input_disp = {
    "input": {
        "img_left": "./data/left.png",
        "img_right": "./data/right.png",
        "no_data": np.nan,
        "disp_min_x": 7,
        "disp_max_x": 2,
        "disp_min_y": -2,
        "disp_max_y": 2,
    }
}
