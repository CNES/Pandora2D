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

from typing import Dict
import numpy as np
import xarray as xr
from json_checker import And, Checker, Or, MissKeyCheckerError, OptionalKey
from rasterio.io import DatasetReader

from pandora.img_tools import get_metadata, rasterio_open
from pandora.check_configuration import (
    check_dataset,
    check_images,
    get_config_input,
    rasterio_can_open_mandatory,
    update_conf,
)

from pandora.check_configuration import rasterio_can_open
from pandora2d.state_machine import Pandora2DMachine


def check_datasets(left: xr.Dataset, right: xr.Dataset) -> None:
    """
    Check that left and right datasets are correct

    :param left: dataset
    :type dataset: xr.Dataset
    :param right: dataset
    :type dataset: xr.Dataset
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
    :type user_cfg: dict
    :param estimation_config: get estimation config if in user_config
    :type estimation_config: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
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
        left_image_metadata = get_metadata(cfg["input"]["left"]["img"])
        check_disparity(left_image_metadata, cfg["input"])

    # test images
    check_images(cfg["input"])

    return cfg


def check_disparity(image_metadata: xr.Dataset, input_cfg: Dict) -> None:
    """
    All checks on disparity

    :param image_metadata: only metadata on the left image
    :type image_metadata: xr.Dataset
    :param input_cfg: input configuration
    :type input_cfg: Dict

    """

    # Check that disparities are dictionaries or grids
    if not (isinstance(input_cfg["row_disparity"], dict) and isinstance(input_cfg["col_disparity"], dict)):
        raise AttributeError("The disparities in rows and columns must be given as 2 dictionaries.")

    if isinstance(input_cfg["row_disparity"]["init"], str) and isinstance(input_cfg["col_disparity"]["init"], str):

        # Read disparity grids
        disparity_row_reader = rasterio_open(input_cfg["row_disparity"]["init"])
        disparity_col_reader = rasterio_open(input_cfg["col_disparity"]["init"])

        # Check disparity grids size and number of bands
        check_disparity_grids(image_metadata, disparity_row_reader)
        check_disparity_grids(image_metadata, disparity_col_reader)

        # Get correct disparity dictionaries from init disparity grids to give as input of
        # the check_disparity_ranges_are_inside_image method
        row_disp_dict = get_dictionary_from_init_grid(disparity_row_reader, input_cfg["row_disparity"]["range"])
        col_disp_dict = get_dictionary_from_init_grid(disparity_col_reader, input_cfg["col_disparity"]["range"])

    elif isinstance(input_cfg["row_disparity"]["init"], int) and isinstance(input_cfg["col_disparity"]["init"], int):
        row_disp_dict = input_cfg["row_disparity"]
        col_disp_dict = input_cfg["col_disparity"]

    else:
        raise ValueError("Initial columns and row disparity values must be two strings or two integers")

    # Check that disparity ranges are not totally out of the image
    check_disparity_ranges_are_inside_image(image_metadata, row_disp_dict, col_disp_dict)


def check_disparity_grids(image_metadata: xr.Dataset, disparity_reader: DatasetReader) -> None:
    """
    Check that disparity grids contains two bands and are
    the same size as the input image

    :param image_metadata:
    :type image_metadata: xr.Dataset
    :param disparity_reader: disparity grids
    :type disparity_reader: rasterio.io.DatasetReader
    """

    # Check that disparity grids are 1-channel grids
    if disparity_reader.count != 1:
        raise AttributeError("Initial disparity grid must be a 1-channel grid")

    # Check that disparity grids are the same size as the input image
    if (disparity_reader.height, disparity_reader.width) != (
        image_metadata.sizes["row"],
        image_metadata.sizes["col"],
    ):
        raise AttributeError("Initial disparity grids and image must have the same size")


def get_dictionary_from_init_grid(disparity_reader: DatasetReader, disp_range: int) -> Dict:
    """
    Get correct dictionaries to give as input of check_disparity_ranges_are_inside_image method
    from initial disparity grids.

    :param disparity_reader: initial disparity grid
    :type disparity_reader: rasterio.io.DatasetReader
    :param disp_range: range of exploration
    :type disp_range: int
    :return: a disparity dictionary to give to check_disparity_ranges_are_inside_image() method
    :rtype: Dict
    """

    init_disp_grid = disparity_reader.read(1)

    # Get dictionary with integer init value corresponding to the maximum absolute value of init_disp_grid
    disp_dict = {
        "init": np.max(np.abs(init_disp_grid)),
        "range": disp_range,
    }

    return disp_dict


def check_disparity_ranges_are_inside_image(
    image_metadata: xr.Dataset, row_disparity: Dict, col_disparity: Dict
) -> None:
    """
    Raise an error if disparity ranges are out off image.

    :param image_metadata:
    :type image_metadata: xr.Dataset
    :param row_disparity:
    :type row_disparity: Dict
    :param col_disparity:
    :type col_disparity: Dict
    :return: None
    :rtype: None
    :raises: ValueError
    """
    if np.abs(row_disparity["init"]) - row_disparity["range"] > image_metadata.sizes["row"]:
        raise ValueError("Row disparity range out of image")
    if np.abs(col_disparity["init"]) - col_disparity["range"] > image_metadata.sizes["col"]:
        raise ValueError("Column disparity range out of image")


def check_roi_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
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
    :type user_cfg: dict
    :param pandora2d_machine: instance of PandoraMachine
    :type pandora2d_machine: PandoraMachine object
    :return: cfg: pipeline configuration
    :rtype: cfg: dict
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

    return pipeline_cfg


def check_expert_mode_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
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
    :type user_cfg: dict
    :param pandora2d_machine: instance of Pandora2DMachine
    :type pandora2d_machine: Pandora2DMachine

    :return: cfg: global configuration
    :rtype: cfg: dict
    """

    # check input
    user_cfg_input = get_config_input(user_cfg)
    cfg_input = check_input_section(user_cfg_input, user_cfg["pipeline"].get("estimation"))

    user_cfg_roi = get_roi_config(user_cfg)
    cfg_roi = check_roi_section(user_cfg_roi)

    # check pipeline
    cfg_pipeline = check_pipeline_section(user_cfg, pandora2d_machine)

    # The estimation step can be utilized independently.
    if "matching_cost" in cfg_pipeline["pipeline"]:
        check_right_nodata_condition(cfg_input, cfg_pipeline)

    output_config = get_output_config(user_cfg)
    check_output_section(output_config)

    cfg_expert_mode = user_cfg.get("expert_mode", {})
    if cfg_expert_mode != {}:
        cfg_expert_mode = check_expert_mode_section(cfg_expert_mode)

    return {**cfg_input, **cfg_roi, **cfg_pipeline, **cfg_expert_mode, "output": output_config}


def get_output_config(user_cfg: Dict) -> Dict:
    """
    Extract output config from user_cfg and fill default values.
    :param user_cfg:
    :type user_cfg:
    :return: output_config
    :rtype: Dict
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
    :type cfg_input: Dict
    :param cfg_pipeline: pipeline section of configuration
    :type cfg_pipeline: Dict
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
    :type roi_cfg: dict
    :param dim: dimension row or col
    :type dim: str
    """
    if roi_cfg["first"] > roi_cfg["last"]:
        raise ValueError('"first" should be lower than "last" in sensor ROI')


def get_roi_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the ROI configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "ROI" in user_cfg:
        cfg["ROI"] = user_cfg["ROI"]

    return cfg


def check_output_section(config: Dict) -> None:
    """
    Validate the given output section.

    :param config: configuration to validate.
    :type config: Dict
    :return: None
    :raise: json_checker errors in the configuration does not respect the schema.
    """
    schema = {"path": str, OptionalKey("format"): And(str, lambda v: v in ["tiff"])}
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
    "col_disparity": {"init": Or(int, rasterio_can_open), "range": And(int, lambda x: x >= 0)},
    "row_disparity": {"init": Or(int, rasterio_can_open), "range": And(int, lambda x: x >= 0)},
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

roi_configuration_schema = {
    "row": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
    "col": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
}

expert_mode_profiling = {"folder_name": str}
