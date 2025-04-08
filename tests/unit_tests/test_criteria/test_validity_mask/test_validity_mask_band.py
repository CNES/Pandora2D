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
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
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
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
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
                "msk_right": np.array(
                    (
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                "step": [3, 2],
                "subpix": 1,
                "window_size": 3,
            },
            np.array(
                [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ]
            ),
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
            np.array(
                (
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                )
            ),
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
            np.array(
                (
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                )
            ),
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
                        [
                            2,
                            2,
                            2,
                            2,
                        ],
                        [
                            2,
                            1,
                            1,
                            1,
                        ],
                        [
                            2,
                            2,
                            1,
                            1,
                        ],
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
                        [
                            2,
                            2,
                            2,
                        ],
                        [
                            2,
                            1,
                            1,
                        ],
                        [
                            2,
                            1,
                            1,
                        ],
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

    cost_volumes = make_cost_volumes

    validity_mask_band = criteria.get_validity_mask_band(cost_volumes["criteria"])

    np.testing.assert_array_equal(validity_mask_band, expected)
