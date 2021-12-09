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

from pandora import read_config_file, read_img, setup_logging
from pandora.common import save_config

from pandora2d import check_json, common
from pandora2d.state_machine import Pandora2DMachine


def run(
    pandora2d_machine: Pandora2DMachine,
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    disp_min_col: int,
    disp_max_col: int,
    disp_min_row: int,
    disp_max_row: int,
    cfg_pipeline: Dict[str, dict],
):
    """
    Run the Pandora 2D pipeline

    :param pandora2d_machine: instance of Pandora2DMachine
    :type pandora2d_machine: Pandora2DMachine
    :param img_left: left Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :type img_left: xarray.Dataset
    :param img_right: right Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :type img_right: xarray.Dataset
    :param disp_min_col: minimal disparity for columns
    :type disp_min_col: int
    :param disp_max_col: maximal disparity for columns
    :type disp_max_col: int
    :param disp_min_row: minimal disparity for lines
    :type disp_min_row: int
    :param disp_max_row: maximal disparity for lines
    :type disp_max_row: int
    :param cfg_pipeline: pipeline configuration
    :type cfg_pipeline: Dict[str, dict]

    :return: None
    """

    pandora2d_machine.run_prepare(img_left, img_right, disp_min_col, disp_max_col, disp_min_row, disp_max_row)

    for e in list(cfg_pipeline):
        pandora2d_machine.run(e, cfg_pipeline)

    pandora2d_machine.run_exit()

    return pandora2d_machine.dataset_disp_maps


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

    pandora2d_machine = Pandora2DMachine()

    cfg = check_json.check_conf(user_cfg, pandora2d_machine)

    setup_logging(verbose)

    # read images
    img_left = read_img(cfg["input"]["img_left"], cfg["input"]["nodata_left"])
    img_right = read_img(cfg["input"]["img_right"], cfg["input"]["nodata_right"])

    ## read disparities values
    disp_min_col = cfg["input"]["disp_min_col"]
    disp_max_col = cfg["input"]["disp_max_col"]
    disp_min_row = cfg["input"]["disp_min_row"]
    disp_max_row = cfg["input"]["disp_max_row"]

    # run pandora 2D and store disp maps in a dataset
    dataset_disp_maps = run(
        pandora2d_machine, img_left, img_right, disp_min_col, disp_max_col, disp_min_row, disp_max_row, cfg["pipeline"]
    )

    # save dataset
    common.save_dataset(dataset_disp_maps, path_output)
    #save config
    save_config(path_output, user_cfg)
