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
"""
Test the refinement.dichotomy pipeline.
"""

import pandora2d


def test_dichotomy_execution_with_spline(left_img_path, right_img_path):
    """Test that execution of Pandora2d with a dichotomy refinement does not fail.

    Uses spline from `scipy.ndimage.map_coordinates`.
    """
    pandora2d_machine = pandora2d.state_machine.Pandora2DMachine()
    user_cfg = {
        "input": {
            "left": {
                "img": str(left_img_path),
                "nodata": "NaN",
            },
            "right": {
                "img": str(right_img_path),
                "nodata": "NaN",
            },
            "col_disparity": [-3, 3],
            "row_disparity": [-3, 3],
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": 7,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
            "refinement": {
                "refinement_method": "dichotomy",
                "iterations": 1,
                "filter": "spline",
            },
        },
    }
    cfg = pandora2d.check_configuration.check_conf(user_cfg, pandora2d_machine)
    image_datasets = pandora2d.img_tools.create_datasets_from_inputs(input_config=cfg["input"])

    pandora2d.run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)
