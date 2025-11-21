# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains methods associated to the pandora2d multiscale mode
"""

import argparse
import logging
from os import PathLike
from pathlib import Path
from typing import Union, Dict
from copy import deepcopy

import rasterio
import xarray as xr
from numpy.typing import NDArray
from pandora import read_config_file
import pandora2d

from pandora2d.common import string_to_path, resolve_path_in_config
from pandora2d import run_pandora2d, run_pandora2d_segment_mode
from pandora2d.state_machine import Pandora2DMachine
from .check_configuration import check_conf, get_tif_shape_list
from .model_estimation import get_init_disparity_grids_with_mesh


# Multiscale pipeline logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    ArgumentParser for multiscale pipeline

    :return parser
    """

    parser = argparse.ArgumentParser(
        description="Run Pandora2D multiscale pipeline",
    )

    parser.add_argument(
        "config_path",
        type=Path,
        help="path to a json file containing the input/output files paths and \
            algorithm parameters for multiscale pipeline",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase output verbosity",
        action="count",
        default=0,
    )

    return parser


def setup_logging(verbose: bool) -> None:
    """
    Setup the logging configuration

    if -v option is given, multiscale pipeline informations are logged
    if -vv option is given, pandora2d pipeline informations are added

    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # Only warnings are logged
    if verbose == 0:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.WARNING)

    # Multiscale pipeline informations are logged
    elif verbose == 1:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    # Multiscale pipeline and pandora2d pipeline informations are logged
    else:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.INFO)
        for name in logging.root.manager.loggerDict:
            if not name.startswith(__name__):
                logging.getLogger(name).setLevel(logging.WARNING)


def resolve_path_in_config_multiscale(config: Dict, config_path: Path) -> Dict:
    """
    Create a copy of config with all path strings replaced by an absolute path string relative to
    config_path.

    :param config: config to modify
    :type config: Dict
    :param config_path: path to the config file.
    :type config_path: Path
    :return: The configuration with changed paths.
    :rtype: Dict
    """
    result = deepcopy(config)
    relative_to = config_path.parent
    result["multiscale"]["left"]["pyramid"] = str(string_to_path(config["multiscale"]["left"]["pyramid"], relative_to))
    result["multiscale"]["right"]["pyramid"] = str(
        string_to_path(config["multiscale"]["right"]["pyramid"], relative_to)
    )

    if left_mask := config["multiscale"]["left"].get("mask"):
        result["multiscale"]["left"]["mask"] = str(string_to_path(left_mask, relative_to))
    if right_mask := config["multiscale"]["right"].get("mask"):
        result["multiscale"]["right"]["mask"] = str(string_to_path(right_mask, relative_to))

    pandora2d_cfg = config["pandora2d"]
    if isinstance(pandora2d_cfg, str):
        result["pandora2d"] = str(string_to_path(pandora2d_cfg, relative_to))
    elif isinstance(pandora2d_cfg, list):
        result["pandora2d"] = [str(string_to_path(json_file, relative_to)) for json_file in pandora2d_cfg]
    result["multiscale"]["output"] = str(string_to_path(config["multiscale"]["output"], relative_to))

    return result


def get_pandora2d_cfg(user_cfg: Dict, path_left_image: Path, path_right_image: Path, resolution_index: int) -> Dict:
    """
    Returns pandora2d configuration for a given resolution to process

    :param user_cfg: user configuration
    :type user_cfg: Dict
    :param path_left_image: path of left image
    :type path_left_image: Path
    :param path_right_image: path of right image
    :type path_right_image: Path
    :param iteration_index: index of the current iteration
    :type iteration_index: int
    :return: pandora2d configuration
    :rtype: Dict
    """

    # Create pandora2d configuration
    if isinstance(user_cfg["pandora2d"], str):
        pandora2d_cfg_path = Path(user_cfg["pandora2d"])
    elif isinstance(user_cfg["pandora2d"], list):
        pandora2d_cfg_path = Path(user_cfg["pandora2d"][resolution_index])
    else:
        raise ValueError("Pandora2d configuration must be a path to a json file or a list of path to json files")

    pandora2d_cfg = read_config_file(pandora2d_cfg_path)
    pandora2d_cfg["input"]["left"]["img"] = path_left_image
    pandora2d_cfg["input"]["right"]["img"] = path_right_image
    pandora2d_cfg = resolve_path_in_config(pandora2d_cfg, pandora2d_cfg_path)

    return pandora2d_cfg


def write_initial_disparity_grid(
    output_path: Path, file_name: Union[Path, str], data: NDArray, dataset_disp_maps: xr.Dataset
) -> None:
    """
    Write initial disparity grid tif file at output_path

    :param output_path: Path to output directory for initial disparity grid
    :type output_path: Path
    :param file_name: file name for initial disparity grid
    :type file_name: Union[Path, str]
    :param data: initial disparity grid
    :type data: NDArray
    :param dataset_disp_maps: computed disparity maps
    :type dataset_disp_maps: xr.Dataset
    """

    output_path.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        str((output_path / str(file_name)).with_suffix(".tif")),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=dataset_disp_maps.attrs["crs"],
        transform=dataset_disp_maps.attrs["transform"],
    ) as dst:
        dst.write(data, 1)


def run_multiscale(config_path: Union[PathLike, str], verbose: bool) -> None:
    """
    Check config file and run multiscale pipeline accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: PathLike|str
    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # Setup logger
    setup_logging(verbose)

    config_path = Path(config_path)

    # read the user input's configuration
    user_cfg = read_config_file(config_path)
    user_cfg = resolve_path_in_config_multiscale(user_cfg, config_path)

    checked_cfg = check_conf(user_cfg)  # pylint: disable=unused-variable

    # Get lists of tif files and their shape
    tif_files_path_left = checked_cfg["multiscale"]["left"]["pyramid"]
    tif_files_path_right = checked_cfg["multiscale"]["right"]["pyramid"]
    tif_files_shape = get_tif_shape_list(tif_files_path_left)

    output_path = checked_cfg["multiscale"]["output"] + "/" + "iteration_"

    for resolution in range(1, len(tif_files_path_left) + 1):

        logger.info("--- Computation for iteration %d ---", resolution)
        logger.info(
            " scale factor = %d between this image and full resolution image",
            tif_files_shape[-1][0] / tif_files_shape[resolution - 1][0],
        )

        pandora2d_cfg = get_pandora2d_cfg(
            checked_cfg, tif_files_path_left[resolution - 1], tif_files_path_right[resolution - 1], resolution - 1
        )

        # We use estimated initial disparity grids computed at the previous resolution
        if resolution != 1:
            pandora2d_cfg["input"]["row_disparity"]["init"] = output_path + str(resolution) + "/init_grid_row.tif"
            pandora2d_cfg["input"]["col_disparity"]["init"] = output_path + str(resolution) + "/init_grid_col.tif"

        pandora2d_machine = Pandora2DMachine()
        checked_pandora2d_cfg = pandora2d.check_configuration.check_conf(pandora2d_cfg, pandora2d_machine)

        # Run pandora2D machine
        if checked_pandora2d_cfg.get("segment_mode", {}).get("enable") is True:
            dataset_disp_maps, completed_cfg = run_pandora2d_segment_mode(pandora2d_machine, checked_pandora2d_cfg)
        else:
            dataset_disp_maps, completed_cfg = run_pandora2d(pandora2d_machine, checked_pandora2d_cfg)

        # We estimate initial disparity grids for next resolution
        if resolution != len(tif_files_path_left):

            # Estimate initial disparity grids for next resolution
            estimated_init_row_grid, estimated_init_col_grid, rmse_row, rmse_col = get_init_disparity_grids_with_mesh(
                dataset_disp_maps, checked_cfg["multiscale"], tif_files_shape[resolution]
            )

            # Save initial disparity grids for next resolution
            output_path_next_res = Path(str(output_path) + str(resolution + 1))
            write_initial_disparity_grid(
                output_path_next_res, "init_grid_row", estimated_init_row_grid, dataset_disp_maps
            )
            write_initial_disparity_grid(
                output_path_next_res, "init_grid_col", estimated_init_col_grid, dataset_disp_maps
            )

            logger.info("RMSE for row disparities is: %f", rmse_row)
            logger.info("RMSE for col disparities is: %f", rmse_col)

        # Save disparity maps
        multiscale_completed_cfg = deepcopy(completed_cfg)
        multiscale_completed_cfg["output"]["path"] = output_path + str(resolution)
        pandora2d.common.save_disparity_maps(dataset_disp_maps, multiscale_completed_cfg)
        logger.info(
            "Disparity maps for iteration %d are saved in %s", resolution, multiscale_completed_cfg["output"]["path"]
        )
        # Save pandora2d configuration
        pandora2d.common.save_config(multiscale_completed_cfg)


def main():
    """
    Call Pandora2D multiscale main
    """

    # Get parser
    parser = get_parser()
    args = parser.parse_args()

    # Run the Pandora 2D pipeline
    run_multiscale(args.config_path, args.verbose)


if __name__ == "__main__":
    main()
