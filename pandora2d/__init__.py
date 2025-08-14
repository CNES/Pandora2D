# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 CS GROUP France
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

from os import PathLike
from pathlib import Path
from typing import Dict, Union, cast
from copy import copy

import xarray as xr

from pandora import read_config_file, setup_logging, import_plugin

from pandora2d import common
from pandora2d.check_configuration import check_conf, check_datasets
from pandora2d.common import string_to_path, resolve_path_in_config
from pandora2d.img_tools import create_datasets_from_inputs, get_roi_processing, remove_roi_margins
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.profiling import generate_summary, expert_mode_config
from pandora2d.margins import Margins, NullMargins
from pandora2d.memory_estimation import segment_image_by_rows


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


def run_pandora2d(pandora2d_machine: Pandora2DMachine, cfg: Dict[str, dict]):
    """
    Process ROI, create image datasets and run pandora2d pipeline

    :param pandora2d_machine: instance of Pandora2DMachine
    :type pandora2d_machine: Pandora2DMachine
    :param cfg: configuration
    :type cfg: Dict[str, dict]
    """

    # check roi in user configuration
    roi = None
    if "ROI" in cfg:
        cfg["ROI"]["margins"] = pandora2d_machine.margins_img.global_margins.astuple()

        # If disparities are computed with estimation step, ROI margins will be updated later
        if "estimation" in cfg["pipeline"]:
            roi = cfg["ROI"]
        else:
            roi = get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])

    # read images
    image_datasets = create_datasets_from_inputs(
        input_config=cfg["input"], roi=roi, estimation_cfg=cfg["pipeline"].get("estimation")
    )

    # check datasets: shape, format and content
    check_datasets(image_datasets.left, image_datasets.right)

    # run pandora 2D and store disp maps in a dataset
    dataset_disp_maps, completed_cfg = run(pandora2d_machine, image_datasets.left, image_datasets.right, cfg)

    # remove ROI margins to save only user ROI in tif files
    if "ROI" in cfg:
        dataset_disp_maps = remove_roi_margins(dataset_disp_maps, cfg)

    return dataset_disp_maps, completed_cfg


def run_pandora2d_segment_mode(pandora2d_machine: Pandora2DMachine, cfg: Dict[str, dict]):
    """
    Run pandora2d pipeline with segment mode

    :param pandora2d_machine: instance of Pandora2DMachine
    :type pandora2d_machine: Pandora2DMachine
    :param cfg: configuration
    :type cfg: Dict[str, dict]
    """

    # Get the list of ROIs to iterate on
    roi_list = segment_image_by_rows(
        cfg, pandora2d_machine.margins_disp.global_margins, pandora2d_machine.margins_img.global_margins
    )

    if not roi_list:
        return run_pandora2d(pandora2d_machine, cfg)

    # Initialisation of output objects
    completed_cfg = {}
    final_dataset_disp_maps = xr.Dataset()
    init_roi = cfg["ROI"] if "ROI" in cfg else None

    # Iteration on the different segments
    for roi in roi_list:

        cfg["ROI"] = cast(Dict, roi)
        dataset_disp_maps, completed_cfg = run_pandora2d(pandora2d_machine, cfg)
        final_dataset_disp_maps = xr.merge([dataset_disp_maps, final_dataset_disp_maps])

    # Add correct ROI in output configuration
    if init_roi is not None:
        completed_cfg["ROI"] = init_roi
        # ROI margins are computed according to disparity and window size
        # so all roi in roi_list have the same margins
        completed_cfg["ROI"]["margins"] = cfg["ROI"]["margins"]
    else:
        del completed_cfg["ROI"]

    # Update offset attribute according to initial ROI
    final_dataset_disp_maps.attrs["offset"] = {
        "row": completed_cfg.get("ROI", {}).get("row", {}).get("first", 0),
        "col": completed_cfg.get("ROI", {}).get("col", {}).get("first", 0),
    }

    return final_dataset_disp_maps, completed_cfg


def main(cfg_path: Union[PathLike, str], verbose: bool) -> None:
    """
    Check config file and run pandora 2D framework accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: PathLike|str
    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # Import pandora plugins
    import_plugin()

    cfg_path = Path(cfg_path)

    # read the user input's configuration
    user_cfg = read_config_file(cfg_path)
    user_cfg = resolve_path_in_config(user_cfg, cfg_path)

    pandora2d_machine = Pandora2DMachine()

    cfg = check_conf(user_cfg, pandora2d_machine)
    expert_mode_config.enable = "expert_mode" in cfg

    setup_logging(verbose)

    if cfg.get("segment_mode", {}).get("enable") is True:
        dataset_disp_maps, completed_cfg = run_pandora2d_segment_mode(pandora2d_machine, cfg)
    else:
        dataset_disp_maps, completed_cfg = run_pandora2d(pandora2d_machine, cfg)

    # save dataset if not empty
    if bool(dataset_disp_maps.data_vars):
        common.save_disparity_maps(dataset_disp_maps, completed_cfg)

    # Update output configuration with detailed margins
    completed_cfg["margins_disp"] = pandora2d_machine.margins_disp.to_dict()
    completed_cfg["margins"] = pandora2d_machine.margins_img.to_dict()
    # save config
    common.save_config(completed_cfg)

    # Profiling results
    if "expert_mode" in completed_cfg:
        path_output = Path(user_cfg["output"]["path"])
        generate_summary(path_output, completed_cfg["expert_mode"]["profiling"])
