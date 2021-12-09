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
from json_checker import Checker, Or, And
import numpy as np

from pandora.check_json import check_disparities, check_images, get_config_input
from pandora.check_json import get_config_pipeline, concat_conf, update_conf, rasterio_can_open_mandatory


from pandora2d.state_machine import Pandora2DMachine


def check_input_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
    """

    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input, user_cfg)

    # check schema
    configuration_schema = {"input": input_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # test images
    check_images(cfg["input"]["img_left"], cfg["input"]["img_right"], None, None)

    # test disparities
    check_disparities(cfg["input"]["disp_min_col"], cfg["input"]["disp_max_col"], None)
    check_disparities(cfg["input"]["disp_min_row"], cfg["input"]["disp_max_row"], None)

    return cfg


def check_pipeline_section(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> Dict[str, dict]:
    """
    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    :param user_cfg: pipeline user configuration
    :type user_cfg: dict
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine object
    :return: cfg: pipeline configuration
    :rtype: cfg: dict
    """

    cfg = update_conf({}, user_cfg)
    pandora2d_machine.check_conf(cfg["pipeline"])

    cfg = update_conf(cfg, pandora2d_machine.pipeline_cfg)

    configuration_schema = {"pipeline": dict}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def check_conf(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> dict:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine
    :return: cfg: global configuration
    :rtype: cfg: dict
    """

    # check input
    user_cfg_input = get_config_input(user_cfg)
    cfg_input = check_input_section(user_cfg_input)

    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, pandora2d_machine)

    if isinstance(cfg_input["input"]["nodata_right"], float) and cfg_pipeline["pipeline"]["matching_cost"][
        "matching_cost_method"
    ] in ["sad", "ssd"]:
        logging.error("nodata_right must be int type with sad or ssd matching_cost_method (ex: 9999)")
        sys.exit(1)

    cfg = concat_conf([cfg_input, cfg_pipeline])

    return cfg


input_configuration_schema = {
    "img_left": And(str, rasterio_can_open_mandatory),
    "img_right": And(str, rasterio_can_open_mandatory),
    "nodata_left": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
    "nodata_right": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
    "disp_min_col": int,
    "disp_max_col": int,
    "disp_min_row": int,
    "disp_max_row": int,
}

default_short_configuration_input = {
    "input": {
        "nodata_left": -9999,
        "nodata_right": -9999,
    }
}
