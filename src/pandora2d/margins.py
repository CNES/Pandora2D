#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Module defining margin handling for Pandora2D.

This module provides classes for representing and manipulating margins.
It includes different types of margins with various configurations.

Classes:
    - Margins: A tuple-like representation of margins with equality comparison.
    - UniformMargins: Margins with the same value in all directions.
    - NullMargins: Margins where all values are set to zero.
"""

from pandora.margins import Margins as BaseMargins


class Margins(BaseMargins):
    """Tuple of margins."""

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.up == other.up and self.down == other.down


class UniformMargins(Margins):
    """Margins set to same values in all directions."""

    def __init__(self, value):
        super().__init__(value, value, value, value)


class NullMargins(UniformMargins):
    """Margins set to zero in all directions."""

    def __init__(self):
        super().__init__(0)
