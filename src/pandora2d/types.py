# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
Objects used as parameters in Pandora2D functions.
"""

from dataclasses import dataclass


@dataclass
class Step:
    """
    Represents the step (distance) between two adjacent pixels in an image
    along the row and column directions.

    Attributes
    ----------
    row : int
        Number of pixels to move along the row direction.
        Default is 1.
    col : int
        Number of pixels to move along the column direction.
        Default is 1.
    """

    row: int = 1
    col: int = 1


@dataclass
class Origin:
    """
    Coordinates used as origin.

    Attributes
    ----------
    row : int
        index along the row direction.
        Default is 0.
    col : int
        index along the column direction.
        Default is 0.
    """

    row: int = 0
    col: int = 0
