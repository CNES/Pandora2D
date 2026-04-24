#
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
This module contains functions allowing to check the configuration given to MVP pipeline.
"""

from os import PathLike
from pathlib import Path
from typing import Dict, Union, List, Tuple

import rasterio

from json_checker import Checker, And, Or
from pandora.check_configuration import update_conf, rasterio_can_open


def tiff_area(file_path: Union[Path, str]) -> int:
    """
    Return tiff area for a given tif file_path

    :param file_path: path to tif file
    :return: tif area width*height
    """
    with rasterio.open(file_path) as src:
        width = src.width
        height = src.height
        return width * height


def get_tif_files_list(path: Path) -> List[Path]:
    """
    Return list of tif files in pyramid repository
    sorted by file area

    :param path: path to pyramid repository
    :return: list of tif files sorted by size (width*height)
    """

    return sorted(path.glob("*.tif"), key=tiff_area)


def get_tif_shape_list(tif_files_list: List[Path]) -> List[Tuple]:
    """
    Return list of tif files shape (height, width)

    :param tif_files_list: list of path to tif files
    :return: list of shape (height, width) for each tif file in tif_files_list
    """

    tif_files_shapes = []

    for path in tif_files_list:
        with rasterio.open(path) as src:
            height = src.height
            width = src.width
            tif_files_shapes.append((height, width))

    return tif_files_shapes


def is_repository_with_tif_file(path: Union[PathLike, str]) -> bool:
    """
    Check if the given path leads to a repository containing at least one readable tif file.

    :param path: path to a repository
    :return: whether the path refers to a valid repository or not
    """

    if path is None:
        return True

    user_path = Path(path)
    if not user_path.is_dir():
        return False

    tif_files = sorted(user_path.glob("*.tif"))

    return len(tif_files) > 0 and all(rasterio_can_open(str(f)) for f in tif_files)


def is_json_file(path: Union[str, Path]) -> bool:
    """
    Check if the given path refers to a JSON file.

    :param path: path to a JSON file
    :return: whether the path refers to a JSON file or not

    """
    user_path = Path(path)

    if not user_path.is_file() or user_path.suffix.lower() != ".json":
        return False

    return True


def check_pyramid_repositories(first_pyramid_path: Union[str, Path], second_pyramid_path: Union[str, Path]):
    """
    Check if first and second pyramid repositories contain the same number of tif files.
    Check if tif files have correct suffix.

    This method is used to check that the left and right image pyramids contain the same number of images,
    and that, if masks are specified, there are as many masks as there are images in the pyramids.

    :param first_pyramid_path: path to first pyramid repository
    :param second_pyramid_path: path to second pyramid repository
    """

    first_pyramid_path = Path(first_pyramid_path)
    second_pyramid_path = Path(second_pyramid_path)

    tif_files_first = get_tif_files_list(first_pyramid_path)
    tif_files_second = get_tif_files_list(second_pyramid_path)

    nb_tif_first = len(tif_files_first)
    nb_tif_second = len(tif_files_second)

    if nb_tif_first != nb_tif_second:
        raise ValueError(
            f"Pyramid repositories '{first_pyramid_path}' ({nb_tif_first} tif files) and "
            f"'{second_pyramid_path}' ({nb_tif_second} tif files) must contain the same number of tif files."
        )


def check_mask_pyramid_repositories(user_cfg: Dict) -> None:
    """
    Check if mask pyramid repository contains the same number of tif files as image pyramid repository.
    Check if tif files have correct suffix.

    :param user_cfg: user configuration
    """

    if user_cfg["multiscale"]["left"]["mask_pyramid"] is not None:
        check_pyramid_repositories(
            user_cfg["multiscale"]["left"]["mask_pyramid"], user_cfg["multiscale"]["left"]["img_pyramid"]
        )
    if user_cfg["multiscale"]["right"]["mask_pyramid"] is not None:
        check_pyramid_repositories(
            user_cfg["multiscale"]["right"]["mask_pyramid"], user_cfg["multiscale"]["right"]["img_pyramid"]
        )


def get_multiscale_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the multiscale configuration

    :param user_cfg: user configuration
    :return cfg: partial configuration
    """

    cfg = {}

    if "multiscale" in user_cfg:
        cfg["multiscale"] = user_cfg["multiscale"]

    return cfg


def check_multiscale_section(user_cfg) -> Dict[str, dict]:
    """
    Check multiscale section of configuration

    :param user_cfg: user configuration
    :return: cfg: checked multiscale configuration
    """

    # Check multiscale configuration
    if "multiscale" not in user_cfg:
        raise KeyError("multiscale key is missing")

    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_configuration_multiscale, user_cfg)

    configuration_schema = {"multiscale": multiscale_configuration_schema}

    # Check schema
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    check_pyramid_repositories(cfg["multiscale"]["left"]["img_pyramid"], cfg["multiscale"]["right"]["img_pyramid"])

    # Check that we have as many input masks as input images if we have mask pyramids
    check_mask_pyramid_repositories(cfg)

    return cfg


def get_pandora2d_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the pandora2d configuration

    :param user_cfg: user configuration
    :return cfg: partial configuration
    """

    cfg = {}

    if "pandora2d" in user_cfg:
        cfg["pandora2d"] = user_cfg["pandora2d"]

    return cfg


def check_pandora2d_section(user_cfg) -> None:
    """
    Check pandora2d section of configuration

    :param user_cfg: user configuration
    :return: cfg: checked pandora2d configuration
    """

    # Check pandora2d configuration
    if "pandora2d" not in user_cfg:
        raise KeyError("pandora2d key is missing")

    configuration_schema = {"pandora2d": pandora2d_configuration_schema}

    # check schema
    checker = Checker(configuration_schema)
    checker.validate(user_cfg)


def update_pyramid_configuration(cfg_multiscale: Dict) -> None:
    """
    Update pyramid configurations with list of tif files sorted by ascending size
    instead of path to pyramid repository.

    :param cfg_multiscale: multiscale configuration
    """

    # Update image pyramid configurations
    cfg_multiscale["multiscale"]["left"]["img_pyramid"] = get_tif_files_list(
        Path(cfg_multiscale["multiscale"]["left"]["img_pyramid"])
    )
    cfg_multiscale["multiscale"]["right"]["img_pyramid"] = get_tif_files_list(
        Path(cfg_multiscale["multiscale"]["right"]["img_pyramid"])
    )
    # Update mask pyramid configurations
    if cfg_multiscale["multiscale"]["left"]["mask_pyramid"] is not None:
        cfg_multiscale["multiscale"]["left"]["mask_pyramid"] = get_tif_files_list(
            Path(cfg_multiscale["multiscale"]["left"]["mask_pyramid"])
        )
    if cfg_multiscale["multiscale"]["right"]["mask_pyramid"] is not None:
        cfg_multiscale["multiscale"]["right"]["mask_pyramid"] = get_tif_files_list(
            Path(cfg_multiscale["multiscale"]["right"]["mask_pyramid"])
        )


def check_conf(user_cfg: Dict) -> Dict[str, dict]:
    """
    Check multiscale configuration

    :param user_cfg: user configuration

    :return: cfg: checked multiscale configuration
    """

    user_cfg_multiscale = get_multiscale_config(user_cfg)
    cfg_multiscale = check_multiscale_section(user_cfg_multiscale)

    cfg_pandora2d = get_pandora2d_config(user_cfg)
    check_pandora2d_section(cfg_pandora2d)

    # Update pyramid configurations with list of tif files sorted by ascending size
    # instead of path to pyramid repository
    update_pyramid_configuration(cfg_multiscale)

    # If we have different pandora2d configurations,
    # we check that we have as many as there are resolutions to process.
    if isinstance(cfg_pandora2d["pandora2d"], list):
        if len(cfg_multiscale["multiscale"]["left"]["img_pyramid"]) != len(cfg_pandora2d["pandora2d"]):
            raise ValueError(
                "If you fill in several pandora2d configuration files, "
                "you must have as many as there are images to process in the pyramid."
            )

    return {**cfg_multiscale, **cfg_pandora2d}


multiscale_configuration_schema = {
    "left": {
        "img_pyramid": And(str, is_repository_with_tif_file),
        "mask_pyramid": And(Or(str, lambda input: input is None), is_repository_with_tif_file),
    },
    "right": {
        "img_pyramid": And(str, is_repository_with_tif_file),
        "mask_pyramid": And(Or(str, lambda input: input is None), is_repository_with_tif_file),
    },
    "model": {"type": And(str, lambda s: s == "pol"), "degree": And(int, lambda d: d >= 0)},
    "mesh": {"row": And(int, lambda x: x > 0), "col": And(int, lambda x: x > 0)},
    "minimal_nb_pixels_per_mesh": And(int, lambda nb: nb > 0),
    "output": str,
}

pandora2d_configuration_schema = Or(
    lambda s: isinstance(s, str) and is_json_file(s), lambda l: isinstance(l, list) and all(is_json_file(x) for x in l)
)


default_configuration_multiscale = {
    "multiscale": {
        "left": {"mask_pyramid": None},
        "right": {"mask_pyramid": None},
        "model": {
            "type": "pol",
            "degree": 2,
        },
        "mesh": {"row": 1, "col": 1},
        "minimal_nb_pixels_per_mesh": 1,
    }
}
