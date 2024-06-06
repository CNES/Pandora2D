#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
from pandora import read_config_file, setup_logging, import_plugin
from pandora.common import save_config

from pandora2d import common
from pandora2d.check_configuration import check_conf, check_datasets
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing
from pandora2d.state_machine import Pandora2DMachine


def run(
    pandora2d_machine: Pandora2DMachine,
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    cfg: Dict[str, dict],
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
    :param cfg: configuration
    :type cfg: Dict[str, dict]

    :return: None
    """

    pandora2d_machine.run_prepare(img_left, img_right, cfg)

    for e in list(cfg["pipeline"]):
        pandora2d_machine.run(e, cfg)

    pandora2d_machine.run_exit()

    return pandora2d_machine.dataset_disp_maps, pandora2d_machine.completed_cfg


def main(cfg_path: str, path_output: str, verbose: bool) -> None:
    """
    Check config file and run pandora 2D framework accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: string
    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # Import pandora plugins
    import_plugin()

    # read the user input's configuration
    user_cfg = read_config_file(cfg_path)

    pandora2d_machine = Pandora2DMachine()

    cfg = check_conf(user_cfg, pandora2d_machine)

    setup_logging(verbose)

    # read disparities values
    col_disparity = cfg["input"]["col_disparity"]
    row_disparity = cfg["input"]["row_disparity"]

    # check roi in user configuration
    roi = None
    if "ROI" in cfg:
        cfg["ROI"]["margins"] = pandora2d_machine.margins.global_margins.astuple()
        roi = get_roi_processing(cfg["ROI"], col_disparity, row_disparity)

    # read images
    image_datasets = create_datasets_from_inputs(
        input_config=cfg["input"], roi=roi, estimation_cfg=cfg["pipeline"].get("estimation")
    )

    # check datasets: shape, format and content
    check_datasets(image_datasets.left, image_datasets.right)

    # run pandora 2D and store disp maps in a dataset
    dataset_disp_maps, completed_cfg = run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

    # save dataset if not empty
    if bool(dataset_disp_maps.data_vars):
        common.save_dataset(dataset_disp_maps, completed_cfg, path_output)
    # Update output configuration with detailed margins
    completed_cfg["margins"] = pandora2d_machine.margins.to_dict()
    # save config
    save_config(path_output, completed_cfg)
