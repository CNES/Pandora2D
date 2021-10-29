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
This module contains functions to run Pandora pipeline.
"""

import pandora

from pandora2d import check_json

def main(cfg_path: str, path_output: str, verbose: bool) -> None:

    """
    Check config file and run pandora 2D framework accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: string
    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # read the user input's configuration
    user_cfg = pandora.read_config_file(cfg_path)

    cfg = check_json.check_input_section(user_cfg)

    #read images
    _ = pandora.read_img(cfg["input"]["img_left"],cfg["input"]["no_data"])

    _ = pandora.read_img(cfg["input"]["img_right"],cfg["input"]["no_data"])




