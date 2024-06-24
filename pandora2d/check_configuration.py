#!/usr/bin/env python
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions allowing to check the configuration given to Pandora pipeline.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import xarray as xr
from json_checker import And, Checker, Or

from pandora.img_tools import get_metadata
from pandora.check_configuration import (
    check_dataset,
    check_disparities_from_input,
    check_images,
    concat_conf,
    get_config_input,
    rasterio_can_open_mandatory,
    update_conf,
)

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

    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input, user_cfg)

    # test disparities
    if estimation_config is not None:
        # add wrong disparity for checking only
        cfg = update_conf(default_configuration_disp, cfg)

    # check schema
    configuration_schema = {"input": input_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # test images
    check_images(cfg["input"])

    if estimation_config is None:
        check_disparities_from_input(cfg["input"]["col_disparity"], None)
        check_disparities_from_input(cfg["input"]["row_disparity"], None)
        left_image_metadata = get_metadata(cfg["input"]["left"]["img"])
        check_disparity_ranges_are_inside_image(
            left_image_metadata, cfg["input"]["row_disparity"], cfg["input"]["col_disparity"]
        )

    return cfg


def check_disparity_ranges_are_inside_image(
    image_metadata: xr.Dataset, row_disparity_range: List, col_disparity_range: List
) -> None:
    """
    Raise an error if disparity ranges are out off image.

    :param image_metadata:
    :type image_metadata: xr.Dataset
    :param row_disparity_range:
    :type row_disparity_range: List
    :param col_disparity_range:
    :type col_disparity_range: List
    :return: None
    :rtype: None
    :raises: ValueError
    """
    if np.abs(row_disparity_range).min() > image_metadata.sizes["row"]:
        raise ValueError("Row disparity range out of image")
    if np.abs(col_disparity_range).min() > image_metadata.sizes["col"]:
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


def check_conf(user_cfg: Dict, pandora2d_machine: Pandora2DMachine) -> dict:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :param pandora2d_machine: instance of PandoraMachine
    :type pandora2d_machine: PandoraMachine
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

    # The refinement step with interpolation method is not allowed with disparity ranges of size lower than 5
    if (
        "refinement" in cfg_pipeline["pipeline"]
        and cfg_pipeline["pipeline"]["refinement"]["refinement_method"] == "interpolation"
    ):
        check_disparity_range_size(cfg_input["input"]["col_disparity"], "Column")
        check_disparity_range_size(cfg_input["input"]["row_disparity"], "Row")

    cfg = concat_conf([cfg_input, cfg_roi, cfg_pipeline])

    return cfg


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


def check_disparity_range_size(disparity: list[int] | str, title: str) -> None:
    """
    Check that disparity ranges with a size < 5 are not allowed for interpolation refinement method.

    :param disparity: disparity range
    :type disparity: list[int] | str
    :param cfg_pipeline: pipeline section of configuration
    :type cfg_pipeline: Dict
    """

    if isinstance(disparity, list):
        if (abs(disparity[1] - disparity[0]) + 1) < 5:
            raise ValueError(
                title + " disparity range with a size < 5 are not allowed with interpolation refinement method"
            )

    if isinstance(disparity, str):
        raise TypeError("Grid disparities are not yet handled by Pandora2D")


def check_roi_coherence(roi_cfg: dict) -> None:
    """
    Check that the first ROI coords are lower than the last.

    :param roi_cfg: user configuration for ROI
    :type roi_cfg: dict
    :param dim: dimension row or col
    :type dim: str
    """
    if roi_cfg["first"] > roi_cfg["last"]:
        raise ValueError('In ROI "first" should be lower than "last" in sensor ROI')


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


input_configuration_schema = {
    "left": {
        "img": And(str, rasterio_can_open_mandatory),
        "nodata": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
    },
    "right": {
        "img": And(str, rasterio_can_open_mandatory),
        "nodata": Or(int, lambda input: np.isnan(input), lambda input: np.isinf(input)),
    },
    "col_disparity": [int, int],
    "row_disparity": [int, int],
}

default_short_configuration_input = {
    "input": {
        "left": {
            "nodata": -9999,
        },
        "right": {
            "nodata": -9999,
        },
    }
}

default_configuration_disp = {"input": {"col_disparity": [-9999, -9995], "row_disparity": [-9999, -9995]}}

roi_configuration_schema = {
    "row": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
    "col": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
}
