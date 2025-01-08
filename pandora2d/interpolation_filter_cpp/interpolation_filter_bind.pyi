# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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

# pylint: skip-file

from typing import List, Tuple
import numpy as np

class Margins:
    up: int
    down: int
    left: int
    right: int

class AbstractFilter:
    def __init__(self) -> None: ...
    def get_coeffs(self, fractional_shift: float) -> np.ndarray: ...
    @staticmethod
    def apply(resampling_area: np.ndarray, row_coeff: np.ndarray, col_coeff: np.ndarray) -> float: ...
    def interpolate(
        self, image: np.ndarray, col_positions: np.ndarray, row_positions: np.ndarray, max_fractional_value: float = ...
    ) -> List: ...
    def get_margins(self) -> Margins: ...

class Bicubic(AbstractFilter): ...

class CardinalSine(AbstractFilter):
    def __init__(self, half_size:int=6, fractional_shift:float=0) -> None: ...
