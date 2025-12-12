#!/usr/bin/env python
#
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
This module contains functions allowing to check the configuration given to Pandora pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import xarray as xr
from json_checker import And, Checker, MissKeyCheckerError, OptionalKey, Or
from pandora.check_configuration import (
    check_dataset,
    check_images,
    get_config_input,
    rasterio_can_open,
    rasterio_can_open_mandatory,
    update_conf,
)
from pandora.img_tools import rasterio_open
from rasterio.io import DatasetReader

from pandora2d.state_machine import Pandora2DMachine
from .common import all_same


def check_datasets(left: xr.Dataset, right: xr.Dataset) -> None:
    """
    Check that left and right datasets are correct

    :param left: dataset
    :param right: dataset
    """

    # Check the dataset content
    check_dataset(left)
    check_dataset(right)

    # Check disparities at least on the left
    if "col_disparity" not in left or "row_disparity" not in left:
        raise ValueError("left dataset must have column and row disparities DataArrays")

    # Check shape
    # check only the rows and columns, the last two elements of the shape
    if left["im"].data.shape[-2:] != right["im"].data.shape[-2:]:
        raise ValueError("left and right datasets must have the same shape")


def check_input_section(user_cfg: Dict[str, dict], estimation_config: dict = None) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :param estimation_config: get estimation config if in user_config
    :return: cfg: global configuration
    """

    if "input" not in user_cfg:
        raise KeyError("input key is missing")

    if estimation_config is not None and (
        ("col_disparity" in user_cfg["input"]) or ("row_disparity" in user_cfg["input"])
    ):
        raise KeyError(
            "When using estimation, "
            "the col_disparity and row_disparity keys must not be given in the configuration file"
        )

    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input, user_cfg)

    configuration_schema = {
        "input": (
            input_configuration_schema | disparity_schema if estimation_config is None else input_configuration_schema
        )
    }

    # check schema
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    if estimation_config is None:
        # test disparities
        check_disparity(cfg["input"])

    # test images
    check_images(cfg["input"])

    return cfg


def check_disparity(input_cfg: Dict) -> None:
    """
    All checks on disparity and resolve disparity grid paths.

    :param input_cfg: input configuration
    """
    # Check that disparities are dictionaries or grids
    if not (isinstance(input_cfg["row_disparity"], dict) and isinstance(input_cfg["col_disparity"], dict)):
        raise AttributeError("The disparities in rows and columns must be given as 2 dictionaries.")

    row_init = input_cfg["row_disparity"]["init"]
    col_init = input_cfg["col_disparity"]["init"]

    if isinstance(row_init, str) and isinstance(col_init, str):
        given_row_path = Path(row_init)
        given_col_path = Path(col_init)
        given_paths = {given_row_path, given_col_path}

        paths_are_dirs = [p.is_dir() for p in given_paths]
        paths_are_files = [p.is_file() for p in given_paths]

        if any(paths_are_dirs) and any(paths_are_files):
            raise ValueError("Directory must not be mixed with file.")

        if not all_same(given_paths) and all(paths_are_dirs):
            raise ValueError("Row and Col disparities must use the same directory.")

        if all(paths_are_dirs):
            input_cfg["attributes"] = load_attributes(given_row_path)

        row_path = given_row_path if given_row_path.is_file() else given_row_path / "row_map.tif"
        col_path = given_col_path if given_col_path.is_file() else given_col_path / "col_map.tif"

        # Resolve and update paths
        input_cfg["row_disparity"]["init"] = str(row_path.resolve())
        input_cfg["col_disparity"]["init"] = str(col_path.resolve())

    elif isinstance(row_init, int) and isinstance(col_init, int):
        row_disp_dict = input_cfg["row_disparity"]
        col_disp_dict = input_cfg["col_disparity"]
        image_reader = rasterio_open(input_cfg["left"]["img"])
        # Check that disparity ranges are not totally out of the image
        check_disparity_ranges_are_inside_image(image_reader.shape, row_disp_dict, col_disp_dict)
    else:
        raise ValueError("Initial columns and row disparity values must be two strings or two integers")


def check_disparity_grids(input_cfg: dict) -> None:
    """
    Check that disparity grids contains two bands and are
    the same size as the input image

    :param input_cfg: input_configuration
    """
    config = input_cfg["input"]
    row_disparity = config["row_disparity"]
    col_disparity = config["col_disparity"]
    image_reader = rasterio_open(config["left"]["img"])
    disparity_readers = rasterio_open(row_disparity["init"]), rasterio_open(col_disparity["init"])

    # Check that disparity grids are 1-channel grids
    if any(r.count != 1 for r in disparity_readers):
        raise AttributeError("Initial disparity grids must be a 1-channel grid")

    if len(shapes := {r.shape for r in disparity_readers}) > 1:  # more than one shape
        raise AttributeError("Initial disparity grids' sizes do not match", shapes)

    # Check that disparity grids are the same size as the input image
    if disparity_readers[0].shape != image_reader.shape:
        raise AttributeError("Initial disparity grids and image must have the same size")

    # Get correct disparity dictionaries from init disparity grids to give as input of
    # the check_disparity_ranges_are_inside_image method
    row_disp_dict = get_dictionary_from_init_grid(disparity_readers[0], row_disparity["range"])
    col_disp_dict = get_dictionary_from_init_grid(disparity_readers[1], col_disparity["range"])

    check_disparity_ranges_are_inside_image(image_reader.shape, row_disp_dict, col_disp_dict)


def get_dictionary_from_init_grid(disparity_reader: DatasetReader, disp_range: int) -> Dict:
    """
    Get correct dictionaries to give as input of check_disparity_ranges_are_inside_image method
    from initial disparity grids.

    :param disparity_reader: initial disparity grid
    :param disp_range: range of exploration
    :return: a disparity dictionary to give to check_disparity_ranges_are_inside_image() method
    """

    init_disp_grid = disparity_reader.read(1)

    # Get dictionary with integer init value corresponding to the maximum absolute value of init_disp_grid
    disp_dict = {
        "init": np.max(np.abs(init_disp_grid)),
        "range": disp_range,
    }

    return disp_dict


def check_disparity_ranges_are_inside_image(
    image_shape: Sequence[int], row_disparity: Dict, col_disparity: Dict
) -> None:
    """
    Raise an error if disparity ranges are out off image.

    :param image_shape: shape of the left image
    :param row_disparity: row disparity configuration
    :param col_disparity: col disparity configuration
    :return: None
    :raises: ValueError
    """
    if np.abs(row_disparity["init"]) - row_disparity["range"] > image_shape[0]:
        raise ValueError("Row disparity range out of image")
    if np.abs(col_disparity["init"]) - col_disparity["range"] > image_shape[1]:
        raise ValueError("Column disparity range out of image")


def check_segment_mode_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the segment mode dictionary is correct

    :param user_cfg: user configuration
    :return: cfg: global configuration
    """

    # Add missing defaults values in user_cfg
    cfg = update_conf(default_segment_mode_configuration, user_cfg)

    # check schema
    configuration_schema = {"segment_mode": segment_mode_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def check_roi_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :return: cfg: global configuration
    """
    if not user_cfg:
        return {}

    # Add missing roi defaults values in user_cfg
    cfg = update_conf({}, user_cfg)

    # check schema
    configuration_schema = {"ROI": roi_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # check ROI configuration coherence
    check_roi_coherence(cfg["ROI"]["col"])
    check_roi_coherence(cfg["ROI"]["row"])

    return cfg


def check_pipeline_section(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> Dict[str, dict]:
    """
    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    :param user_cfg: pipeline user configuration
    :param pandora2d_machine: instance of PandoraMachine
    :return: cfg: pipeline configuration
    """

    cfg = update_conf({}, user_cfg)

    if "pipeline" not in cfg:
        raise KeyError("pipeline key is missing")

    pandora2d_machine.check_conf(cfg)

    cfg = update_conf(cfg, pandora2d_machine.pipeline_cfg)

    configuration_schema = {"pipeline": dict}

    checker = Checker(configuration_schema)

    # We select only the pipeline section for the checker
    pipeline_cfg = {"pipeline": cfg["pipeline"]}

    checker.validate(pipeline_cfg)

    if "refinement" in pipeline_cfg["pipeline"]:
        check_subpix_value_with_dichotomy(
            pipeline_cfg["pipeline"]["refinement"]["refinement_method"],
            pipeline_cfg["pipeline"]["matching_cost"]["subpix"],
        )

    return pipeline_cfg


def check_step_from_attributes(attributes: dict, expected_step_value: list[int]) -> None:
    """
    Validate that the initial disparity attributes match the pipeline configuration.

    :param disparity_directory: Input section of user configuration dictionary.
    :type disparity_directory: dict
    :param expected_step_value: Checked pipeline section configuration dictionary.
    :type expected_step_value: dict
    :raises AttributeError: If the steps do not match.
    """

    attributes_step = [attributes["step"]["row"], attributes["step"]["col"]]

    if attributes_step != expected_step_value:
        raise AttributeError(
            f"Initial disparity grid step {attributes_step} does not match configuration step {expected_step_value}."
        )


def load_attributes(disparity_directory: Path) -> dict:
    """
    Load attributes from json file in disparity directory.

    :param disparity_directory: directory where to find attributes' file.
    :return: attributes dictionary
    """
    with disparity_directory.joinpath("attributes.json").open(encoding="utf-8") as fd:
        attributes = json.load(fd)
    return attributes


def check_subpix_value_with_dichotomy(refinement_method: str, subpix: int) -> None:
    """
    Check if we have a subpix value of 1 with a dichotomy refinement method,
    in which case we return a warning to prevent aliasing.

    :param refinement_method: refinement method in user configuration
    :param subpix: subpix value in user configuration
    """

    if (refinement_method in ("dichotomy", "dichotomy_python")) and (subpix == 1):
        logging.warning(
            "To avoid aliasing, it is strongly recommended to set the subpix parameter of the matching cost step"
            " to a value greater than 1 when using dichotomy."
        )


def check_expert_mode_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :return: cfg: global configuration
    """

    if "profiling" not in user_cfg:
        raise MissKeyCheckerError("Please be sure to set the profiling dictionary")

    # check profiling schema
    profiling_mode_cfg = user_cfg["profiling"]
    checker = Checker(expert_mode_profiling)
    checker.validate(profiling_mode_cfg)

    profiling_mode_cfg = {"expert_mode": user_cfg}

    return profiling_mode_cfg


def check_conf(user_cfg: Dict, pandora2d_machine: Pandora2DMachine) -> dict:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :param pandora2d_machine: instance of Pandora2DMachine

    :return: cfg: global configuration
    """

    # check input
    user_cfg_input = get_config_input(user_cfg)
    estimation_config = user_cfg["pipeline"].get("estimation")
    cfg_input = check_input_section(user_cfg_input, estimation_config)

    user_cfg_roi = get_roi_config(user_cfg)
    cfg_roi = check_roi_section(user_cfg_roi)

    user_cfg_segment_mode = get_segment_mode_config(user_cfg)
    cfg_segment_mode = check_segment_mode_section(user_cfg_segment_mode)

    # check pipeline
    cfg_pipeline = check_pipeline_section(user_cfg, pandora2d_machine)
    if attributes := cfg_input["input"].get("attributes"):
        check_step_from_attributes(attributes, cfg_pipeline["pipeline"]["matching_cost"]["step"])

    row_init = user_cfg_input["input"].get("row_disparity", {}).get("init")
    if isinstance(row_init, str):
        check_disparity_grids(cfg_input)

    # The estimation step can be utilized independently.
    if "matching_cost" in cfg_pipeline["pipeline"]:
        check_right_nodata_condition(cfg_input, cfg_pipeline)

    output_config = get_output_config(user_cfg)
    check_output_section(output_config)

    cfg_expert_mode = user_cfg.get("expert_mode", {})
    if cfg_expert_mode != {}:
        cfg_expert_mode = check_expert_mode_section(cfg_expert_mode)

    return {**cfg_input, **cfg_segment_mode, **cfg_roi, **cfg_pipeline, **cfg_expert_mode, "output": output_config}


def get_output_config(user_cfg: Dict) -> Dict:
    """
    Extract output config from user_cfg and fill default values.
    :param user_cfg: user configuration
    :return: output_config
    """
    defaults = {"format": "tiff"}
    try:
        config = user_cfg["output"]
    except KeyError:
        raise MissKeyCheckerError("Configuration file is missing output key")
    return {**defaults, **config}


def check_right_nodata_condition(cfg_input: Dict, cfg_pipeline: Dict) -> None:
    """
    Check that only int is accepted for nodata of right image when matching_cost_method is sad or ssd.
    :param cfg_input: inputs section of configuration
    :param cfg_pipeline: pipeline section of configuration
    """

    if not isinstance(cfg_input["input"]["right"]["nodata"], int) and cfg_pipeline["pipeline"]["matching_cost"][
        "matching_cost_method"
    ] in ["sad", "ssd"]:
        raise ValueError(
            "nodata of right image must be of type integer with sad or ssd matching_cost_method (ex: 9999)"
        )


def check_roi_coherence(roi_cfg: dict) -> None:
    """
    Check that the first ROI coords are lower than the last.

    :param roi_cfg: user configuration for ROI
    :param dim: dimension row or col
    """
    if roi_cfg["first"] > roi_cfg["last"]:
        raise ValueError('"first" should be lower than "last" in sensor ROI')


def get_section_config(user_cfg: Dict[str, dict], key: str) -> Dict[str, dict]:
    """
    Get the section configuration from key

    :param user_cfg: user configuration
    :return cfg: partial configuration
    """

    cfg = {}

    if key in user_cfg:
        cfg[key] = user_cfg[key]

    return cfg


def get_segment_mode_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the segment_mode configuration

    :param user_cfg: user configuration
    :return cfg: partial configuration
    """

    return get_section_config(user_cfg, "segment_mode")


def get_roi_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the ROI configuration

    :param user_cfg: user configuration
    :return cfg: partial configuration
    """

    return get_section_config(user_cfg, "ROI")


def check_output_section(config: Dict) -> None:
    """
    Validate the given output section.

    :param config: configuration to validate.
    :return: None
    :raise: json_checker errors in the configuration does not respect the schema.
    """
    schema = {
        "path": str,
        OptionalKey("format"): And(str, lambda v: v in ["tiff"]),
        OptionalKey("deformation_grid"): {"init_pixel_conv_grid": Or([0, 0], [0.5, 0.5])},
    }

    checker = Checker(schema)
    checker.validate(config)


input_configuration_schema = {
    "left": {
        "img": And(str, rasterio_can_open_mandatory),
        "nodata": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
        "mask": And(Or(str, lambda input: input is None), rasterio_can_open),
    },
    "right": {
        "img": And(str, rasterio_can_open_mandatory),
        "nodata": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
        "mask": And(Or(str, lambda input: input is None), rasterio_can_open),
    },
}

disparity_schema = {
    "col_disparity": {"init": Or(int, str), "range": And(int, lambda x: x >= 0)},
    "row_disparity": {"init": Or(int, str), "range": And(int, lambda x: x >= 0)},
}

default_short_configuration_input = {
    "input": {
        "left": {
            "nodata": -9999,
            "mask": None,
        },
        "right": {
            "nodata": -9999,
            "mask": None,
        },
    }
}

default_segment_mode_configuration = {
    "segment_mode": {
        "enable": False,
        "memory_per_work": 1000,
    },
}

segment_mode_configuration_schema = {
    "enable": bool,
    "memory_per_work": And(int, lambda x: x > 0),
}

roi_configuration_schema = {
    "row": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
    "col": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
}

expert_mode_profiling = {"folder_name": str}
