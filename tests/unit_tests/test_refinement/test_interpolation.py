# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
Test refinement step
"""

# pylint: disable=redefined-outer-name, protected-access
# mypy: disable-error-code=attr-defined

import numpy as np
import pytest
import xarray as xr
import json_checker

from pandora.margins import Margins

from pandora2d import common, refinement


@pytest.fixture()
def cv_dataset():
    """
    Create dataset cost volumes
    """

    cv = np.zeros((3, 3, 5, 5))
    cv[:, :, 2, 2] = np.ones([3, 3])
    cv[:, :, 2, 3] = np.ones([3, 3])
    cv[:, :, 3, 2] = np.ones([3, 3])
    cv[:, :, 3, 3] = np.ones([3, 3])

    c_row = np.arange(cv.shape[0])
    c_col = np.arange(cv.shape[1])

    # First pixel in the image that is fully computable (aggregation windows are complete)
    row = np.arange(c_row[0], c_row[-1] + 1)
    col = np.arange(c_col[0], c_col[-1] + 1)

    disparity_range_col = np.arange(-2, 2 + 1)
    disparity_range_row = np.arange(-2, 2 + 1)

    cost_volumes_test = xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_col", "disp_row"], cv)},
        coords={"row": row, "col": col, "disp_col": disparity_range_col, "disp_row": disparity_range_row},
    )

    cost_volumes_test.attrs["measure"] = "zncc"
    cost_volumes_test.attrs["window_size"] = 1
    cost_volumes_test.attrs["type_measure"] = "max"

    return cost_volumes_test


def test_checkconf_fails_if_iterations_is_given():
    """
    Description : Test fails if iterations is given
    Data :
    Requirement : EX_CONF_08
    """
    with pytest.raises(json_checker.core.exceptions.MissKeyCheckerError):
        refinement.interpolation.Interpolation({"refinement_method": "interpolation", "iterations": 1})


def test_margins():
    """
    test margins of matching cost pipeline
    """
    _refinement = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]

    assert _refinement.margins == Margins(3, 3, 3, 3), "Not a cubic kernel Margins"


def test_refinement_method_subpixel(cv_dataset):
    """
    test refinement_method with interpolation
    """

    cost_volumes_test = cv_dataset

    data = np.full((3, 3), 0.4833878)

    data_variables = {
        "row_map": (("row", "col"), data),
        "col_map": (("row", "col"), data),
        "correlation_score": (("row", "col"), data),
    }

    coords = {"row": np.arange(3), "col": np.arange(3)}

    dataset = xr.Dataset(data_variables, coords)

    dataset_disp_map = common.dataset_disp_maps(
        dataset.row_map, dataset.col_map, dataset.coords, dataset.correlation_score
    )

    test = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
    delta_x, delta_y, _ = test.refinement_method(cost_volumes_test, dataset_disp_map, None, None)

    np.testing.assert_allclose(data, delta_y, rtol=1e-06)
    np.testing.assert_allclose(data, delta_x, rtol=1e-06)


def test_refinement_method_pixel(cv_dataset):
    """
    test refinement
    """

    cost_volumes_test = cv_dataset

    new_cv_datas = np.zeros((3, 3, 5, 5))
    new_cv_datas[:, :, 1, 3] = np.ones([3, 3])

    cost_volumes_test["cost_volumes"].data = new_cv_datas

    gt_delta_y = np.array(
        ([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
        dtype=np.float64,
    )

    gt_delta_x = np.array(
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        dtype=np.float64,
    )

    data_variables = {
        "row_map": (("row", "col"), gt_delta_y),
        "col_map": (("row", "col"), gt_delta_x),
        "correlation_score": (("row", "col"), gt_delta_x),
    }

    coords = {"row": np.arange(3), "col": np.arange(3)}

    dataset = xr.Dataset(data_variables, coords)

    dataset_disp_map = common.dataset_disp_maps(
        dataset.row_map, dataset.col_map, dataset.coords, dataset.correlation_score
    )

    test = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
    delta_x, delta_y, _ = test.refinement_method(cost_volumes_test, dataset_disp_map, None, None)

    np.testing.assert_allclose(gt_delta_y, delta_y, rtol=1e-06)
    np.testing.assert_allclose(gt_delta_x, delta_x, rtol=1e-06)


def test_refinement_correlation_score(cv_dataset):
    """
    test correlation_score with interpolation
    """

    cost_volumes_test = cv_dataset

    data = np.full((3, 3), 1.33731468)

    data_variables = {
        "row_map": (("row", "col"), data),
        "col_map": (("row", "col"), data),
        "correlation_score": (("row", "col"), data),
    }

    coords = {"row": np.arange(3), "col": np.arange(3)}

    dataset = xr.Dataset(data_variables, coords)

    dataset_disp_map = common.dataset_disp_maps(
        dataset.row_map, dataset.col_map, dataset.coords, dataset.correlation_score
    )

    test = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
    _, _, correlation_score = test.refinement_method(cost_volumes_test, dataset_disp_map, None, None)

    np.testing.assert_allclose(data, correlation_score, rtol=1e-06)
