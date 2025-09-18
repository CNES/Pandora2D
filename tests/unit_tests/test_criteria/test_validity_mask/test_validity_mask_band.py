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
Test that the validity mask band obtained with the criteria dataarray is correct.
"""

import numpy as np
import pytest

from pandora2d import criteria
from pandora2d.constants import Criteria
from pandora2d.margins import Margins


@pytest.mark.parametrize(
    ["make_cost_volumes", "expected"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="No mask and window_size=1",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 1, 1, 1, 1, 1, 1, 2],
                    [2, 1, 1, 0, 0, 1, 1, 2],
                    [2, 1, 1, 0, 0, 1, 1, 2],
                    [2, 1, 1, 1, 1, 1, 1, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="No mask and window_size=3",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [2, 3],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2],
                    [2, 0, 1],
                    [2, 1, 1],
                ]
            ),
            id="No mask, window_size=3, step=[2,3]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 5,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 1, 1, 1, 1, 2, 2],
                    [2, 2, 1, 1, 1, 1, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="No mask and window_size=5",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 2, 0, 0, 2, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 2, 0, 0, 0, 1, 1],
                    [1, 1, 0, 2, 0, 2, 1, 2],
                    [1, 1, 1, 1, 1, 2, 1, 1],
                ]
            ),
            id="Left mask, window_size=1",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 1, 2],
                    [2, 2, 2, 2, 0, 1, 1, 2],
                    [2, 1, 2, 2, 2, 1, 2, 2],
                    [2, 1, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="Left mask, window_size=3",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [3, 2],
                "subpix": 1,
                "window_size": 3,
            },
            np.full((2, 4), 2),
            id="Left mask, window_size=3, step=[3,2]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            np.full((6, 8), 1),
            id="Right mask, window_size=1",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                # Temporarily, the criteria for subpixel disparities
                # are raised by following a nearest neighbor strategy.
                "subpix": 2,
                "window_size": 1,
            },
            np.full((6, 8), 1),
            id="Right mask, window_size=1, subpix=2",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                )
            ),
            id="Right mask, window_size=3",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [2, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                    ]
                )
            ),
            id="Right mask, window_size=3, step=[2,1]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 2],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2],
                        [2, 1, 1, 1],
                        [2, 1, 1, 1],
                        [2, 2, 1, 1],
                        [2, 2, 1, 1],
                        [2, 2, 2, 2],
                    ]
                )
            ),
            id="Right mask, window_size=3, step=[1,2]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [2, 2],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2],
                        [2, 1, 1, 1],
                        [2, 2, 1, 1],
                    ]
                )
            ),
            id="Right mask, window_size=3, step=[2,2]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [2, 3],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2],
                        [2, 1, 1],
                        [2, 1, 1],
                    ]
                )
            ),
            id="Right mask, window_size=3, step=[2,3]",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                # Temporarily, the criteria for subpixel disparities
                # are raised by following a nearest neighbor strategy.
                "subpix": 2,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                )
            ),
            id="Right mask, window_size=3, subpix=2",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                # Temporarily, the criteria for subpixel disparities
                # are raised by following a nearest neighbor strategy.
                "subpix": 4,
                "window_size": 3,
            },
            np.array(
                (
                    [
                        [2, 2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 1, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 1, 1, 1, 1, 2],
                        [2, 2, 2, 2, 2, 2, 2, 2],
                    ]
                )
            ),
            id="Right mask, window_size=3, subpix=4",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 1, 2],
                    [2, 2, 2, 2, 1, 1, 1, 2],
                    [2, 2, 2, 2, 2, 1, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="Left and right masks, window_size=3",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                # Temporarily, the criteria for subpixel disparities
                # are raised by following a nearest neighbor strategy.
                "subpix": 2,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 1, 2],
                    [2, 2, 2, 2, 1, 1, 1, 2],
                    [2, 2, 2, 2, 2, 1, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="Left and right masks, window_size=3, subpix=2",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 2, 2, 2, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [1, 1],
                # Temporarily, the criteria for subpixel disparities
                # are raised by following a nearest neighbor strategy.
                "subpix": 4,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 1, 2],
                    [2, 2, 2, 2, 1, 1, 1, 2],
                    [2, 2, 2, 2, 2, 1, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="Left and right masks, window_size=3, subpix=4",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_validity_mask(make_cost_volumes, expected):
    """
    Test that the produced validity mask bands are correct according to:
        - disparities
        - window size
        - left and right masks
    """

    cost_volumes = make_cost_volumes()

    # subset with real user disparities
    subset = cost_volumes["criteria"].sel(
        disp_row=slice(cost_volumes.attrs["row_disparity_source"][0], cost_volumes.attrs["row_disparity_source"][1]),
        disp_col=slice(cost_volumes.attrs["col_disparity_source"][0], cost_volumes.attrs["col_disparity_source"][1]),
    )

    validity_mask_band = criteria.get_validity_mask_band(subset)

    np.testing.assert_array_equal(validity_mask_band, expected)


@pytest.mark.parametrize(
    ["make_cost_volumes", "refinement_margins", "reset_first_point", "expected"],
    [
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 2},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            Margins(1, 1, 2, 2),
            False,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            id="No mask, subpix=1, step=[1,1], window_size=1 and bicubic margins",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 2},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 1,
                "window_size": 1,
            },
            Margins(6, 6, 6, 6),
            False,
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 2, 1, 1, 2, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 2, 0, 0, 0, 1, 1],
                    [1, 1, 1, 2, 1, 2, 1, 2],
                    [1, 1, 1, 1, 1, 2, 1, 1],
                ]
            ),
            id="Left mask, subpix=1, step=[1,1], window_size=1 and cardinal sine margins",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 2},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [1, 1],
                "subpix": 2,
                "window_size": 3,
            },
            Margins(6, 6, 6, 6),
            True,
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 1, 2],
                    [2, 2, 2, 2, 1, 1, 1, 2],
                    [2, 1, 2, 2, 2, 1, 2, 2],
                    [2, 1, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
            id="Left mask, subpix=2, step=[1,1], window_size=3 and cardinal sine margins and one element margins valid",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "msk_right": np.full((6, 8), 0),
                "step": [3, 2],
                "subpix": 1,
                "window_size": 3,
            },
            Margins(3, 1, 1, 1),
            True,
            np.full((2, 4), 2),
            id="Left mask, window_size=3, step=[3,2] and random margins and one element on margins valid",
        ),
        pytest.param(
            {
                "row_disparity": {"init": 0, "range": 1},
                "col_disparity": {"init": 0, "range": 2},
                "msk_left": np.full((6, 8), 0),
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 2, 2, 2, 2, 2],
                            [0, 0, 0, 1, 0, 2, 0, 1],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                        ]
                    )
                ),
                "step": [3, 2],
                "subpix": 4,
                "window_size": 3,
            },
            Margins(13, 1, 1, 21),
            True,
            np.array(
                (
                    [
                        [2, 2, 2, 2],
                        [2, 1, 1, 1],
                    ]
                )
            ),
            id="Right mask, window_size=3, step=[3,2] and random margins and one element on margins valid",
        ),
    ],
    indirect=["make_cost_volumes"],
)
def test_validity_mask_with_refinement(make_cost_volumes, refinement_margins, reset_first_point, expected):
    """
    Test that the produced validity mask bands are correct according to:
        - disparities
        - refinement method (margins added to cost_volume)
    """

    cost_volumes_without_margins = make_cost_volumes()

    # subset with real user disparities
    row_disparity = cost_volumes_without_margins.attrs["row_disparity_source"]
    col_disparity = cost_volumes_without_margins.attrs["col_disparity_source"]
    subset_without_margins = cost_volumes_without_margins["criteria"].sel(
        disp_row=slice(row_disparity[0], row_disparity[1]),
        disp_col=slice(col_disparity[0], col_disparity[1]),
    )

    validity_mask_band_without_margins = criteria.get_validity_mask_band(subset_without_margins)

    cost_volumes_with_margins = make_cost_volumes(refinement_margins)
    if reset_first_point:
        # The first refinement_margins.left columns are set to Criteria.VALID to verify that margins are not
        # taken into account.
        cost_volumes_with_margins["criteria"].data[0, 0, :, : refinement_margins.left] = Criteria.VALID

    row_disparity = cost_volumes_with_margins.attrs["row_disparity_source"]
    col_disparity = cost_volumes_with_margins.attrs["col_disparity_source"]
    subset_with_margins = cost_volumes_with_margins["criteria"].sel(
        disp_row=slice(row_disparity[0], row_disparity[1]),
        disp_col=slice(col_disparity[0], col_disparity[1]),
    )
    validity_mask_band_with_margins = criteria.get_validity_mask_band(subset_with_margins)

    np.testing.assert_array_equal(validity_mask_band_without_margins, expected)
    np.testing.assert_array_equal(validity_mask_band_with_margins, expected)
