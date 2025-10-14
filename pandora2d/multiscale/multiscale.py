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

import sys
from os import PathLike
from pathlib import Path
from typing import Union, Dict
from copy import deepcopy
from pandora import read_config_file
from pandora2d.common import string_to_path
from .check_configuration import check_conf


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

    result["pandora2d"] = str(string_to_path(config["pandora2d"], relative_to))
    result["multiscale"]["output"] = str(string_to_path(config["multiscale"]["output"], relative_to))

    return result


def main(config_path: Union[PathLike, str]) -> None:
    """
    Check config file and run multiscale pipeline accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: PathLike|str
    :return: None
    """

    config_path = Path(config_path)

    # read the user input's configuration
    user_cfg = read_config_file(config_path)
    user_cfg = resolve_path_in_config_multiscale(user_cfg, config_path)

    checked_cfg = check_conf(user_cfg)  # pylint: disable=unused-variable


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    main(cfg_path)
