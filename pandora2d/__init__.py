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

from typing import Dict
import xarray as xr

from pandora import read_config_file, read_img

from pandora2d import check_json
from pandora2d.state_machine import Pandora2DMachine


def run(
    pandora2d_machine: Pandora2DMachine,
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    disp_min_x: int,
    disp_max_x: int,
    disp_min_y: int,
    disp_max_y: int,
    cfg_pipeline: Dict[str, dict],
):
    """
    Run the Pandora 2D pipeline

    Args:
        pandora2d_machine (Pandora2DMachine): instance of Pandora2DMachine
        img_left (xr.Dataset): left Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
        img_right (xr.Dataset): right Dataset image containing :

              - im : 2D (row, col) xarray.DataArray
              - msk (optional): 2D (row, col) xarray.DataArray
        disp_min_x (int): minimal disparity for columns
        disp_max_x (int): maximal disparity for columns
        disp_min_y (int): minimal disparity for lines
        disp_max_y (int): maximal disparity for lines
        cfg_pipeline (Dict[str, dict]): pipeline configuration

    Returns:
        int: [description]
    """

    pandora2d_machine.run_prepare(img_left, img_right, disp_min_x, disp_max_x, disp_min_y, disp_max_y)

    for e in list(cfg_pipeline):
        pandora2d_machine.run(e, cfg_pipeline)
        if pandora2d_machine.state == "begin":
            break

    pandora2d_machine.run_exit()


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
    user_cfg = read_config_file(cfg_path)

    cfg = check_json.check_input_section(user_cfg)

    pandora2d_machine = Pandora2DMachine()

    cfg = check_json.check_conf(user_cfg, pandora2d_machine)

    # read images
    img_left = read_img(cfg["input"]["img_left"], cfg["input"]["no_data"])

    img_right = read_img(cfg["input"]["img_right"], cfg["input"]["no_data"])

    ## read disparities values

    disp_min_x = cfg["input"]["disp_min_x"]
    disp_max_x = cfg["input"]["disp_max_x"]
    disp_min_y = cfg["input"]["disp_min_y"]
    disp_max_y = cfg["input"]["disp_max_y"]

    run(pandora2d_machine, img_left, img_right, disp_min_x, disp_max_x, disp_min_y, disp_max_y, cfg["pipeline"])
