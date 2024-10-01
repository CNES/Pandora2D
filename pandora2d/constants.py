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
"""
This module contains all the parameters related to the criteria dataset, defining each bit.
"""

from enum import Flag, auto


class Criteria(Flag):
    """
    Criteria class
    """

    VALID = 0

    PANDORA2D_MSK_PIXEL_LEFT_BORDER = auto()
    """The pixel is invalid : border of left image according to window size."""
    PANDORA2D_MSK_PIXEL_LEFT_NODATA = auto()
    """The pixel is invalid : nodata in left mask."""
    PANDORA2D_MSK_PIXEL_RIGHT_NODATA = auto()
    """The pixel is invalid : nodata in right mask."""
    PANDORA2D_MSK_PIXEL_RIGHT_DISPARITY_OUTSIDE = auto()
    """The pixel is invalid : disparity is out the right image."""
    PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_LEFT = auto()
    """The pixel is invalid : invalidated by validity mask of left image."""
    PANDORA2D_MSK_PIXEL_INVALIDITY_MASK_RIGHT = auto()
    """The pixel is invalid : invalidated by validity mask of right image."""
    PANDORA2D_MSK_PIXEL_PEAK_ON_EDGE = auto()
    """
    The pixel is invalid : The correlation peak is at the edge of disparity range.
    The calculations stopped at the pixellic stage.
    """
    PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED = auto()
    """The disparity is not processed because not included in the disparity range of the current point."""
