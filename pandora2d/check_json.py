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
This module contains functions allowing to check the configuration given to Pandora pipeline.
"""
import sys
from typing import Dict
import logging
import numpy as np

from pandora.check_json import check_disparities, check_images, get_config_input, get_config_pipeline, concat_conf

from pandora2d.state_machine import Pandora2DMachine


def check_input_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Check if the input dictionary is correct

    Args:
        user_cfg (Dict[str, dict]): user pipeline configuration

    Returns:
        Dict[str, dict]: global configuration
    """

    # test images
    check_images(user_cfg["input"]["img_left"], user_cfg["input"]["img_right"], None, None)

    if "no_data" in user_cfg["input"]:
        if user_cfg["input"]["no_data"] is None:
            user_cfg["input"]["no_data"] = np.nan
    else:
        logging.error("no_data must be initialized")
        sys.exit(1)

    # test disparities
    check_disparities(user_cfg["input"]["disp_min_x"], user_cfg["input"]["disp_max_x"], None)
    check_disparities(user_cfg["input"]["disp_min_y"], user_cfg["input"]["disp_max_y"], None)

    return user_cfg


def check_pipeline_section(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> Dict[str, dict]:
    """

    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    Args:
        user_cfg (Dict[str, dict]): user pipeline configuration
        pandora2d_machine (Pandora2DMachine): instance of Pandora2DMachine

    Returns:
        Dict[str, dict]: pipeline configuration
    """

    pandora2d_machine.check_conf(user_cfg["pipeline"])

    return user_cfg


def check_conf(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> dict:
    """
    Check if dictionnary is correct

    Args:
        user_cfg (Dict[str, dict]): pipeline configuration
        pandora2d_machine (Pandora2DMachine): instance of Pandora2DMachine

    Returns:
        dict: global configuration
    """

    # check input
    user_cfg_input = get_config_input(user_cfg)
    cfg_input = check_input_section(user_cfg_input)

    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, pandora2d_machine)

    cfg = concat_conf([cfg_input, cfg_pipeline])

    return cfg
