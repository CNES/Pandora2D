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

# pylint: skip-file
import numpy as np
from numpy.typing import NDArray
from typing import overload

class CostVolumeSize:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, cv_size: NDArray[np.floating]) -> None: ...
    @overload
    def __init__(self, _r: int, _c: int, _dr: int, _dc: int) -> None: ...
    def size(self) -> int:
        """
        Returns the cost_volume size nb_row * nb_col * nb_disp_row * nb_disp_col.
        """

    def nb_disps(self) -> int:
        """
        Returns the disparity number : nb_disp_row * nb_disp_col.
        """
