#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the interpolation method used in the refinement step.
"""

from typing import Dict, Union
from pandora.check_json import update_conf
from json_checker import And, Checker

from . import refinement


@refinement.AbstractRefinement.register_subclass("interpolation")
class Interpolation(refinement.AbstractRefinement):
    """
    Interpolation class allows to perform the subpixel cost refinement step
    """

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        self.cfg = self.check_conf(**cfg)

    @staticmethod
    def check_conf(**cfg: Dict[str, Union[str, int]]) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Check the refinement configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        cfg = update_conf({"refinement_method": "interpolation"}, cfg)

        schema = {
            "refinement_method": And(str, lambda x: x in ["interpolation"]),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg
