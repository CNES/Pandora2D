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
"""Module with global performance test fixtures and methods."""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name

from typing import Tuple

import pytest

import numpy as np

from numpy.typing import NDArray


@pytest.fixture()
def remove_edges():
    """
    Remove medicis disparity maps edges
    """

    def inner(
        medicis_map: NDArray[np.floating], pandora2d_map: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get reduced disparity maps after removing medicis edges full of nans (greater than pandora2d edges)
        on both pandora2d and medicis disparity maps.
        """

        # Gets coordinates for which medicis col_map is different from nan
        # i.e. points that are not within the edges
        non_nan_row_indexes, non_nan_col_indexes = np.where(~np.isnan(medicis_map))

        # Remove medicis edges
        medicis_map = medicis_map[
            non_nan_row_indexes[0] : non_nan_row_indexes[-1] + 1, non_nan_col_indexes[0] : non_nan_col_indexes[-1] + 1
        ]

        # Remove pandora2d edges to get the same points as the ones in medicis disparity maps
        pandora2d_map = pandora2d_map[
            non_nan_row_indexes[0] : non_nan_row_indexes[-1] + 1, non_nan_col_indexes[0] : non_nan_col_indexes[-1] + 1
        ]

        return medicis_map, pandora2d_map

    return inner


@pytest.fixture()
def data_path(root_dir):
    """
    Return path to get left and right images and medicis data
    """
    return root_dir / "tests/performance_tests/data_medicis/"


@pytest.fixture()
def shift_path(data_path, img_path):
    """
    Return path to get left and right images and medicis data
    """
    return data_path / img_path


@pytest.fixture()
def medicis_maps_path(shift_path, medicis_method_path):
    """
    Return path to get medicis data
    """
    return shift_path / medicis_method_path
