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
This module contains functions allowing to check the configuration given to Pandora2d pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import xarray as xr
from json_checker import And, Checker, OptionalKey, Or
from pandora.check_configuration import (
    check_dataset,
    check_images,
    get_metadata,
    rasterio_can_open,
    rasterio_can_open_mandatory,
    update_conf,
)
from pandora.img_tools import rasterio_open
from rasterio.io import DatasetReader

from pandora2d.state_machine import Pandora2DMachine
from pandora2d.common import all_same


def check_datasets(left: xr.Dataset, right: xr.Dataset) -> None:
    """
    Check that left and right datasets are correct

    :param left: dataset
    :param right: dataset
    :raises ValueError: If required disparities are missing or dataset shapes differ.
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


def check_conf(user_cfg: Dict, pandora2d_machine: Pandora2DMachine) -> dict:
    """
    Validate and complete the user configuration.

    :param user_cfg: user configuration dictionary
    :param pandora2d_machine: Pandora2DMachine instance

    :return: global configuration
    """

    # Check sections without dependencies
    check_segment_mode_section(user_cfg)
    check_pipeline_section(user_cfg, pandora2d_machine)
    check_output_section(user_cfg)
    check_expert_mode_section(user_cfg)

    # Check section with dependencies
    # The input section must be checked after the pipeline because it depends on the matching_cost step value
    estimation_config = user_cfg["pipeline"].get("estimation")
    check_input_section(user_cfg, estimation_config)

    # The roi section must be checked after the input section because disparity grids can define a ROI
    check_roi_section(user_cfg)

    # Check nodata and matching_cost method
    # The nodata value must be checked after the input section because the parameter is optional.
    if "matching_cost" in user_cfg["pipeline"]:
        check_right_nodata_condition(user_cfg["input"], user_cfg["pipeline"])

    return user_cfg


def get_section_config(user_cfg: Dict[str, dict], key: str) -> Dict[str, dict]:
    """
    Get the section configuration from key

    :param user_cfg: user configuration dictionary
    :param key: section name
    :return cfg: configuration section dictionary or empty dict
    """

    cfg = {}

    if key in user_cfg:
        cfg[key] = user_cfg[key]

    return cfg


def update_global_conf(global_cfg: Dict[str, dict], completed_cfg: Dict[str, dict]) -> None:
    """
    Update global_cfg with completed_cfg

    :param global_cfg: configuration to be updated
    :param completed_cfg: configuration used for the update
    """
    for key, value in completed_cfg.items():
        if isinstance(value, Mapping):
            if key not in global_cfg or not isinstance(global_cfg.get(key), Mapping):
                global_cfg[key] = {}
            update_global_conf(global_cfg[key], value)
        else:
            if value == "NaN":
                global_cfg[key] = np.nan
            elif value == "inf":
                global_cfg[key] = np.inf
            elif value == "-inf":
                global_cfg[key] = -np.inf
            else:
                global_cfg[key] = value


def check_segment_mode_section(user_cfg: Dict[str, dict]) -> None:
    """
    Complete and check if the segment mode dictionary is correct

    :param user_cfg: user configuration dictionary
    """

    # Get segment mode config
    user_cfg_segment_mode = get_section_config(user_cfg, "segment_mode")

    # Add missing defaults values in user_cfg
    cfg = build_default_segment_mode_configuration()
    update_global_conf(cfg, user_cfg_segment_mode)

    # Check schema
    configuration_schema = {"segment_mode": segment_mode_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    update_global_conf(user_cfg, cfg)


def check_pipeline_section(user_cfg: Dict[str, dict], pandora2d_machine: Pandora2DMachine) -> None:
    """
    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    :param user_cfg: user configuration dictionary
    :param pandora2d_machine: Pandora2DMachine instance
    :raises KeyError: If the pipeline section is missing
    """

    # Check pipeline key
    if "pipeline" not in user_cfg:
        raise KeyError("pipeline key is missing")

    # Converted NaN and inf strings to numpy values
    user_cfg_pipeline = update_conf({}, user_cfg)

    # Check all step on state machine
    pandora2d_machine.check_conf(user_cfg_pipeline)

    update_global_conf(user_cfg, pandora2d_machine.pipeline_cfg)

    # Check subpix value with dichotomy
    if "refinement" in user_cfg["pipeline"]:
        check_subpix_value_with_dichotomy(
            user_cfg["pipeline"]["refinement"]["refinement_method"],
            user_cfg["pipeline"]["matching_cost"]["subpix"],
        )


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


def check_output_section(user_cfg: Dict[str, dict]) -> None:
    """
    Validate the given output section.

    :param user_cfg: user configuration dictionary
    """

    # Get output configuration
    user_cfg_output = get_section_config(user_cfg, "output")

    # Check schema
    configuration_schema = {"output": output_configuration_schema}
    checker = Checker(configuration_schema)
    checker.validate(user_cfg_output)

    update_global_conf(user_cfg, user_cfg_output)


def check_expert_mode_section(user_cfg: Dict[str, dict]) -> None:
    """
    Check if expert mode section is correct

    :param user_cfg: user configuration dictionary
    """

    # Get expert mode config
    user_cfg_expert_mode = get_section_config(user_cfg, "expert_mode")

    if user_cfg_expert_mode:
        # Check schema
        configuration_schema = {"expert_mode": expert_mode_profiling_schema}
        checker = Checker(configuration_schema)
        checker.validate(user_cfg_expert_mode)

    update_global_conf(user_cfg, user_cfg_expert_mode)


def check_input_section(user_cfg: Dict[str, dict], estimation_config: dict = None) -> None:
    """
    Complete and check if the input is correct

    :param user_cfg: user configuration dictionary
    :param estimation_config: get estimation config if in user_config
    :raises KeyError: If the input section is missing or incompatible with estimation mode
    """

    if "input" not in user_cfg:
        raise KeyError("input key is missing")

    # Get input section config
    user_cfg_input = get_section_config(user_cfg, "input")

    if estimation_config is not None and (
        ("col_disparity" in user_cfg_input["input"]) or ("row_disparity" in user_cfg_input["input"])
    ):
        raise KeyError(
            "When using estimation, "
            "the col_disparity and row_disparity keys must not be given in the configuration file"
        )

    # Add missing steps and inputs defaults values in user_cfg
    input_cfg = build_default_short_configuration_input()
    update_global_conf(input_cfg, user_cfg_input)

    configuration_schema = {
        "input": (
            input_configuration_schema | disparity_schema if estimation_config is None else input_configuration_schema
        )
    }

    # check schema
    checker = Checker(configuration_schema)
    checker.validate(input_cfg)

    if estimation_config is None:
        # test disparities
        left_image_metadata = get_metadata(input_cfg["input"]["left"]["img"])
        check_disparity(left_image_metadata, input_cfg["input"], user_cfg)

    # test images
    check_images(input_cfg["input"])

    update_global_conf(user_cfg, input_cfg)


def check_disparity(image_metadata: xr.Dataset, input_cfg: Dict, user_cfg: Dict) -> None:
    """
    All checks on disparity

    :param image_metadata: left image metadata
    :param input_cfg: input configuration with default value
    :param user_cfg: user configuration dictionary
    :raises AttributeError: If disparity definitions or grids are invalid
    :raises ValueError: If disparity ranges are inconsistent with the image
    """

    # Check that disparities are dictionaries or grids
    if not (isinstance(input_cfg["row_disparity"], dict) and isinstance(input_cfg["col_disparity"], dict)):
        raise AttributeError("The disparities in rows and columns must be given as 2 dictionaries.")

    row_init = input_cfg["row_disparity"]["init"]
    col_init = input_cfg["col_disparity"]["init"]

    # row_init & col_init can be files or a pandora2d output directory from a previous run
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

        # Get path
        row_path = given_row_path if given_row_path.is_file() else given_row_path / "row_map.tif"
        col_path = given_col_path if given_col_path.is_file() else given_col_path / "col_map.tif"

        # Resolve and update paths
        input_cfg["row_disparity"]["init"] = str(row_path.resolve())
        input_cfg["col_disparity"]["init"] = str(col_path.resolve())

        # Read disparity grids
        disparity_row_reader = rasterio_open(input_cfg["row_disparity"]["init"])
        disparity_col_reader = rasterio_open(input_cfg["col_disparity"]["init"])

        # Check disparity grids size and number of bands
        check_disparity_grids(image_metadata, disparity_row_reader, disparity_col_reader, given_row_path, user_cfg)

        # Get correct disparity dictionaries from init disparity grids to give as input of
        # the check_disparity_ranges_are_inside_image method
        row_disp_dict = get_dictionary_from_init_grid(disparity_row_reader, input_cfg["row_disparity"]["range"])
        col_disp_dict = get_dictionary_from_init_grid(disparity_col_reader, input_cfg["col_disparity"]["range"])

    # row_init & col_init have a single common value for all pixels
    elif isinstance(row_init, int) and isinstance(col_init, int):
        row_disp_dict = input_cfg["row_disparity"]
        col_disp_dict = input_cfg["col_disparity"]

    else:
        raise ValueError("Initial columns and row disparity values must be two strings or two integers")

    # Check that disparity ranges are not totally out of the image
    check_disparity_ranges_are_inside_image(image_metadata, row_disp_dict, col_disp_dict)


def check_disparity_grids(
    image_metadata: xr.Dataset,
    disparity_row_reader: DatasetReader,
    disparity_col_reader: DatasetReader,
    row_path: Path,
    user_cfg: Dict,
) -> None:
    """
    Check that disparity grids contains two bands and are the same size as the input image

    :param image_metadata: left image metadata
    :param disparity_row_reader: row disparity raster reader
    :param disparity_col_reaser: col disparity raster reader
    :param row_path: disparity file or directory path
    :param user_cfg: user configuration dictionary
    :raises AttributeError: If grid dimensions, bands, or attributes are invalid
    """
    disparity_readers = disparity_row_reader, disparity_col_reader

    # Check that disparity grids are 1-channel grids
    if any(r.count != 1 for r in disparity_readers):
        raise AttributeError("Initial disparity grids must be a 1-channel grid")

    # Check shape is the same for the two grids
    if len(shapes := {r.shape for r in disparity_readers}) > 1:  # more than one shape
        raise AttributeError("Initial disparity grids' sizes do not match", shapes)

    # Check disparity grids are inside image
    # input_cfg["row_disparity"]["init"] &  input_cfg["col_disparity"]["init"] = directory
    if row_path.is_dir():

        # Load attributes parameter
        attributes = load_attributes(row_path)

        # Check step attributes
        check_step_from_attributes(attributes, user_cfg["pipeline"]["matching_cost"]["step"])

        # Check that the disparity grid size is <= the image size and lies within the image bounds
        new_roi = check_disparity_grids_from_directory_within_image(attributes, disparity_row_reader, image_metadata)

        if new_roi:
            if "ROI" in user_cfg:
                logging.warning(
                    "The ROI given in the user configuration will be replaced by the ROI derived from the disparity"
                    " grids."
                )
            user_cfg["ROI"] = new_roi

        # Update user configuration
        user_cfg["attributes"] = attributes

    # Check that disparity grids are the same size as the input image
    elif (disparity_row_reader.height, disparity_row_reader.width) != (
        image_metadata.sizes["row"],
        image_metadata.sizes["col"],
    ):
        raise AttributeError("Initial disparity grids and image must have the same size")


def load_attributes(disparity_directory: Path) -> dict:
    """
    Load attributes from json file in disparity directory.

    :param disparity_directory: directory where to find attributes' file.
    :return: attributes dictionary
    """
    with disparity_directory.joinpath("attributes.json").open(encoding="utf-8") as fd:
        attributes = json.load(fd)
    return attributes


def check_step_from_attributes(attributes: dict, expected_step_value: list[int]) -> None:
    """
    Validate that the initial disparity attributes match the pipeline configuration.

    :param attributes: dictionnary grid attributes
    :param expected_step_value: expected step values.
    :raises AttributeError: If the steps do not match.
    """

    attributes_step = [attributes["step"]["row"], attributes["step"]["col"]]

    if attributes_step != expected_step_value:
        raise AttributeError(
            f"Initial disparity grid step {attributes_step} does not match configuration step {expected_step_value}."
        )


def check_disparity_grids_from_directory_within_image(
    attributes: dict, disparity_row_reader: DatasetReader, image_metadata: xr.Dataset
) -> Union[dict, None]:
    """
    Check that disparity grids lie within image boundaries.

    :param attributes: dictionnary grid attributes
    :param disparity_row_reader: row disparity raster reader
    :param image_metadata: left image metadata
    :return: ROI dictionary if grids define a sub-area, otherwise None
    :raises AttributeError: If disparity grids exceed image boundaries
    """

    # Get row coordinates
    row_min = attributes["origin_coordinates"]["row"]
    row_max = row_min + disparity_row_reader.height * attributes["step"]["row"]

    # Get col coordinates
    col_min = attributes["origin_coordinates"]["col"]
    col_max = col_min + disparity_row_reader.width * attributes["step"]["col"]

    image_height, image_width = image_metadata.sizes["row"], image_metadata.sizes["col"]
    if not (row_min >= 0 and col_min >= 0 and row_max <= image_height and col_max <= image_width):
        raise AttributeError("Initial disparity grid is not inside image boundaries.")

    if row_max < image_height or col_max < image_width:
        return update_roi_from_disparity_grid(row_min, row_max, col_min, col_max)

    return None


def update_roi_from_disparity_grid(row_min: int, row_max: int, col_min: int, col_max: int) -> dict:
    """
    Construct ROI from input disparity grids when there are smaller than image,

    :param row_min: minimum row index
    :param row_max: maximum row index (exclusive)
    :param col_min: minimum col index
    :param col_max: maximum col index (exclusive)
    :return: ROI dictionary
    """

    return {
        "row": {
            "first": row_min,
            "last": row_max - 1,
        },
        "col": {
            "first": col_min,
            "last": col_max - 1,
        },
    }


def get_dictionary_from_init_grid(disparity_reader: DatasetReader, disp_range: int) -> Dict:
    """
    Get correct dictionaries to give as input of check_disparity_ranges_are_inside_image method
    from initial disparity grids.

    :param disparity_reader: disparity grid reader
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
    image_metadata: xr.Dataset, row_disparity: Dict, col_disparity: Dict
) -> None:
    """
    Raise an error if disparity ranges are out off image.

    :param image_metadata: left image metadata
    :param row_disparity: row disparity configuration
    :param col_disparity: column disparity configuration
    :raises ValueError: If ranges exceed image bounds
    """
    if np.abs(row_disparity["init"]) - row_disparity["range"] > image_metadata.sizes["row"]:
        raise ValueError("Row disparity range out of image")
    if np.abs(col_disparity["init"]) - col_disparity["range"] > image_metadata.sizes["col"]:
        raise ValueError("Column disparity range out of image")


def check_roi_section(user_cfg: Dict[str, dict]) -> None:
    """
    Complete and check if roi section is correct

    :param user_cfg: user configuration dictionary
    """

    # Get roi config
    user_cfg_roi = get_section_config(user_cfg, "ROI")

    if user_cfg_roi:
        # check schema
        configuration_schema = {"ROI": roi_configuration_schema}
        checker = Checker(configuration_schema)
        checker.validate(user_cfg_roi)

        # check ROI configuration coherence
        check_roi_coherence(user_cfg_roi["ROI"]["col"])
        check_roi_coherence(user_cfg_roi["ROI"]["row"])

    update_global_conf(user_cfg, user_cfg_roi)


def check_roi_coherence(roi_cfg: dict) -> None:
    """
    Check that the first ROI coords are lower than the last.

    :param roi_cfg: user configuration for ROI
    :raises ValueError: If first coordinate is greater than last
    """
    if roi_cfg["first"] > roi_cfg["last"]:
        raise ValueError('"first" should be lower than "last" in sensor ROI')


def check_right_nodata_condition(cfg_input: Dict, cfg_pipeline: Dict) -> None:
    """
    Check that only int is accepted for nodata of right image when matching_cost_method is sad or ssd.
    :param cfg_input: inputs section of configuration
    :param cfg_pipeline: pipeline section of configuration
    :raises ValueError: If nodata type is invalid
    """

    if not isinstance(cfg_input["right"]["nodata"], int) and cfg_pipeline["matching_cost"]["matching_cost_method"] in [
        "sad",
        "ssd",
    ]:
        raise ValueError(
            "nodata of right image must be of type integer with sad or ssd matching_cost_method (ex: 9999)"
        )


def build_default_short_configuration_input() -> dict:
    """Default configuration input"""
    return {
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


def build_default_segment_mode_configuration() -> dict:
    """Default segment mode"""
    return {
        "segment_mode": {
            "enable": False,
            "memory_per_work": 1000,
        },
    }


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

segment_mode_configuration_schema = {
    "enable": bool,
    "memory_per_work": And(int, lambda x: x > 0),
}

roi_configuration_schema = {
    "row": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
    "col": {"first": And(int, lambda x: x >= 0), "last": And(int, lambda x: x >= 0)},
}

expert_mode_profiling_schema = {
    "profiling": {"folder_name": str},
}

output_configuration_schema = {
    "path": str,
    OptionalKey("format"): And(str, lambda v: v in ["tiff"]),
    OptionalKey("deformation_grid"): {"init_pixel_conv_grid": Or([0, 0], [0.5, 0.5])},
}
