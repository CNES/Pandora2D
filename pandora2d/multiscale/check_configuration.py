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
from typing import Dict, Union
from json_checker import Checker, And, Or
from pandora.check_configuration import update_conf, rasterio_can_open


def is_repository_with_tif_file(path: Union[PathLike, str]) -> bool:
    """
    Check if the given path leads to a repository containing at least one readable tif file.

    :param path: path to a repository
    :type path: Union[PathLike, str]
    :return: whether the path refers to a valid repository or not
    :rtype: bool
    """

    user_path = Path(path)
    if not user_path.is_dir():
        return False

    tif_files = list(user_path.glob("*.tif"))

    return len(tif_files) > 0 and all(rasterio_can_open(str(f)) for f in tif_files)


def is_json_file(path: Union[str, Path]) -> bool:
    """
    Check if the given path refers to a JSON file.

    :param path: path to a JSON file
    :type path: Union[str, Path]
    :return: whether the path refers to a JSON file or not
    :rtype: bool
    """
    user_path = Path(path)

    if not user_path.is_file() or user_path.suffix.lower() != ".json":
        return False

    return True


def get_multiscale_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the multiscale configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "multiscale" in user_cfg:
        cfg["multiscale"] = user_cfg["multiscale"]

    return cfg


def check_multiscale_section(user_cfg) -> Dict[str, dict]:
    """
    Check multiscale section of configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: checked multiscale configuration
    :rtype: cfg: dict
    """

    # Check multiscale configuration
    if "multiscale" not in user_cfg:
        raise KeyError("multiscale key is missing")

    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_configuration_multiscale, user_cfg)

    configuration_schema = {"multiscale": multiscale_configuration_schema}

    # check schema
    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def get_pandora2d_config(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the pandora2d configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "pandora2d" in user_cfg:
        cfg["pandora2d"] = user_cfg["pandora2d"]

    return cfg


def check_pandora2d_section(user_cfg) -> None:
    """
    Check pandora2d section of configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: checked multiscale configuration
    :rtype: cfg: dict
    """

    # Check pandora2d configuration
    if "pandora2d" not in user_cfg:
        raise KeyError("pandora2d key is missing")

    configuration_schema = {"pandora2d": pandora2d_configuration_schema}

    # check schema
    checker = Checker(configuration_schema)
    checker.validate(user_cfg)


def check_conf(user_cfg: Dict) -> Dict[str, dict]:
    """
    Check multiscale configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: checked multiscale configuration
    :rtype: cfg: dict
    """

    user_cfg_multiscale = get_multiscale_config(user_cfg)
    cfg_multiscale = check_multiscale_section(user_cfg_multiscale)

    cfg_pandora2d = get_pandora2d_config(user_cfg)
    check_pandora2d_section(cfg_pandora2d)

    return {**cfg_multiscale, **cfg_pandora2d}


multiscale_configuration_schema = {
    "left": {"pyramid": And(str, is_repository_with_tif_file), "mask": Or(None, And(str, is_repository_with_tif_file))},
    "right": {
        "pyramid": And(str, is_repository_with_tif_file),
        "mask": Or(None, And(str, is_repository_with_tif_file)),
    },
    "model": {"type": And(str, lambda s: s == "pol"), "degree": And(int, lambda d: d >= 0)},
    "output": str,
}

pandora2d_configuration_schema = And(str, is_json_file)

default_configuration_multiscale = {
    "multiscale": {
        "left": {"mask": None},
        "right": {"mask": None},
        "model": {
            "type": "pol",
            "degree": 2,
        },
    }
}
