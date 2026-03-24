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
Test ambiguity cost volume confidence method
"""

import copy
import json_checker
import numpy as np
import pytest
import xarray as xr

from pandora2d import cost_volume_confidence
from pandora2d.img_tools import add_disparity_grid
from pandora2d.margins import Margins

# pylint: disable=redefined-outer-name, protected-access


@pytest.fixture()
def ambiguity_cfg():
    return {"confidence_method": "ambiguity"}


@pytest.fixture()
def cost_volume_confidence_object(ambiguity_cfg):
    return cost_volume_confidence.CostVolumeConfidenceRegistry.get(ambiguity_cfg["confidence_method"])


class TestFactory:  # pylint: disable=too-few-public-methods
    """
    Test instances of CostVolumeConfidence
    """

    def test_factory_ambiguity(self, ambiguity_cfg, cost_volume_confidence_object):
        """
        Test instance of CostVolumeConfidence with ambiguity method
        """

        cost_volume_confidence_instance = cost_volume_confidence_object(ambiguity_cfg)

        assert isinstance(cost_volume_confidence_instance, cost_volume_confidence.CostVolumeConfidence)
        assert isinstance(cost_volume_confidence_instance, cost_volume_confidence.Ambiguity)


class TestCheckConf:
    """
    Test check configuration of ambiguity method
    """

    def test_check_conf(self, ambiguity_cfg):
        """
        Test check_conf of ambiguity method
        """
        cost_volume_confidence.Ambiguity(ambiguity_cfg)

    def test_default_values(self, ambiguity_cfg, cost_volume_confidence_object):
        """
        Test default values of ambiguity method
        """

        cost_volume_confidence_instance = cost_volume_confidence_object(ambiguity_cfg)

        assert cost_volume_confidence_instance._method == "ambiguity"
        assert cost_volume_confidence_instance._eta_max == 0.7
        assert cost_volume_confidence_instance._eta_step == 0.01
        assert cost_volume_confidence_instance._normalization is True

    @pytest.mark.parametrize(
        ["wrong_ambiguity_cfg"],
        [
            pytest.param(
                {"confidence_method": "wrong_ambiguity"},
                id="Wrong method name",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 2},
                id="Eta max out of bounds",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": -4},
                id="Eta step out of bounds",
            ),
            pytest.param(
                {"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.02, "normalization": "not_bool"},
                id="Normalization is not a boolean",
            ),
        ],
    )
    def test_fails_with_incorrect_cfg(self, cost_volume_confidence_object, wrong_ambiguity_cfg):
        """
        Test that check conf fails when the configuration is incorrect
        """

        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            cost_volume_confidence_object(wrong_ambiguity_cfg)

    def test_fails_with_missing_confidence_method_key(self, cost_volume_confidence_object):
        """
        Test that check conf fails when confidence key is missing in the configuration
        """

        with pytest.raises(
            json_checker.core.exceptions.MissKeyCheckerError,
            match="Missing keys in current response: confidence_method",
        ):
            cost_volume_confidence_object({"eta_max": 0.2, "eta_step": 0.02})


@pytest.fixture
def row():
    """The number of rows"""
    return 4


@pytest.fixture
def col():
    """The number of columns"""
    return 4


@pytest.fixture
def shape(row, col):
    """Image shape"""
    return (row, col)


@pytest.fixture
def row_disparity():
    """Default row disparity configuration"""
    return {"init": 1, "range": 2}


@pytest.fixture
def col_disparity():
    """Default column disparity configuration"""
    return {"init": -1, "range": 2}


@pytest.fixture
def subpix():
    """Default subpix"""
    return 1


@pytest.fixture
def margins():
    """Default cost_volume margins"""
    return Margins(0, 0, 0, 0)


@pytest.fixture
def disps_row(row_disparity, margins, subpix):
    """Range of row disparity"""
    disp_min = row_disparity["init"] - row_disparity["range"] - margins.up
    disp_max = row_disparity["init"] + row_disparity["range"] + margins.down
    return np.arange(disp_min, disp_max + 1, 1 / float(subpix))


@pytest.fixture
def disps_col(col_disparity, margins, subpix):
    """Range of column disparity"""
    disp_min = col_disparity["init"] - col_disparity["range"] - margins.left
    disp_max = col_disparity["init"] + col_disparity["range"] + margins.right
    return np.arange(disp_min, disp_max + 1, 1 / float(subpix))


@pytest.fixture()
def left_datasets(row, col, row_disparity, col_disparity):
    """
    Creates left datasets
    """
    left = xr.Dataset(
        {"im": (["row", "col"], np.full((row, col), 1))},
        coords={"row": np.arange(row), "col": np.arange(col)},
    )

    return add_disparity_grid(left, col_disparity, row_disparity)


@pytest.fixture
def cost_volume_init_value():
    """Default initial value for cost_volume"""
    return np.nan


@pytest.fixture
def cost_volume(row, col, disps_row, disps_col, cost_volume_init_value, subpix):
    """Create a cost_volume"""
    np_data = np.full((row, col, len(disps_row), len(disps_col)), cost_volume_init_value, dtype=float)

    return xr.Dataset(
        {"cost_volumes": (["row", "col", "disp_row", "disp_col"], np_data)},
        coords={"row": np.arange(row), "col": np.arange(col), "disp_row": disps_row, "disp_col": disps_col},
        attrs={"subpixel": subpix, "type_measure": "max"},
    )


@pytest.fixture()
def dataset_disp_maps(row, col):
    """Empty dataset_disp_maps"""
    return xr.Dataset(
        coords={
            "row": np.arange(row),
            "col": np.arange(col),
        }
    )


@pytest.fixture()
def empty_dataset():
    """
    Empty dataset to check that the warning is printed when the confidence_prediction method is called.
    Fixture to be deleted when the ambiguity has been implemented.
    """
    return xr.Dataset()


class TestConfidencePrediction:
    """
    Test confidence_prediction method
    """

    @pytest.fixture()
    def cost_volume_confidence_instance(self, cost_volume_confidence_object, ambiguity_cfg):
        """Create ambiguity instance"""
        ambiguity_cfg["normalization"] = False
        return cost_volume_confidence_object(ambiguity_cfg)

    @pytest.fixture()
    def nbr_etas(self, cost_volume_confidence_instance):
        """Number of etas"""
        return np.arange(
            cost_volume_confidence_instance._eta_min,
            cost_volume_confidence_instance._eta_max,
            cost_volume_confidence_instance._eta_step,
        ).shape[0]

    @pytest.fixture()
    def expected_value_with_monotonic_surface(self, nbr_etas, cost_volume):
        """Compute expected ambiguity value for monotonic surface"""
        # default value computed if norm_extremum parameter is nan on ambiguity.cpp pandora file
        nbr_disparities = cost_volume.sizes["disp_row"] * cost_volume.sizes["disp_col"]
        # return 1 - ambiguity
        return 1 - (nbr_etas * nbr_disparities)

    @pytest.mark.parametrize("cost_volume_init_value", [np.nan, np.inf, 0, -99, 0.1])
    def test_with_monotonic_surface(
        self,
        cost_volume_confidence_instance,
        empty_dataset,
        left_datasets,
        cost_volume,
        dataset_disp_maps,
        expected_value_with_monotonic_surface,
    ):
        """
        Test confidence_prediction method with monotonic surface
        i.e. cost_surface is filled only with the same value = cost_volume_init_value
        Tests run without normalization
        """

        _, dataset_disp_maps = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume,
            dataset_disp_maps=dataset_disp_maps,
        )

        assert "confidence_measure" in dataset_disp_maps.data_vars
        assert np.all(dataset_disp_maps["confidence_measure"].values == expected_value_with_monotonic_surface)

    @pytest.fixture()
    def expected_value_with_one_peak(self, nbr_etas):
        """Compute expected ambiguity value for one peak"""
        # return 1 - ambiguity
        return 1 - nbr_etas

    @pytest.mark.parametrize("cost_volume_init_value", [0])
    def test_with_one_peak(
        self,
        cost_volume_confidence_instance,
        empty_dataset,
        left_datasets,
        cost_volume,
        dataset_disp_maps,
        expected_value_with_one_peak,
    ):
        """
        Test confidence_prediction method with monotonic surface
        i.e. cost_surface is filled only with one peak (value = 1) and the remaining elements to cost_volume_init_value
        Tests run without normalization
        """

        # For all points, there is only one peak where row = 0 and col = 0
        cost_volume["cost_volumes"].values[:, :, 1, 3] = 1

        _, dataset_disp_maps = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume,
            dataset_disp_maps=dataset_disp_maps,
        )

        assert "confidence_measure" in dataset_disp_maps.data_vars
        assert np.all(dataset_disp_maps["confidence_measure"].values == expected_value_with_one_peak)


class TestNormalizeWithExtremum:
    """
    Test normalize_with_extremum method
    """

    @pytest.fixture()
    def cost_volume_confidence_instance(self, cost_volume_confidence_object, ambiguity_cfg):
        """Create ambiguity instance"""
        return cost_volume_confidence_object(ambiguity_cfg)

    @pytest.mark.parametrize("cost_volume_init_value", [np.nan, np.inf, 0, -99, 0.1])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test_with_monotonic_surface(
        self,
        cost_volume_confidence_instance,
        empty_dataset,
        left_datasets,
        cost_volume,
        dataset_disp_maps,
    ):
        """
        Test confidence_prediction method with monotonic surface
        i.e. cost_surface is filled only with the same value = cost_volume_init_value
        In this case, there is no confidence.
        """

        _, dataset_disp_maps = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume,
            dataset_disp_maps=dataset_disp_maps,
        )

        assert "confidence_measure" in dataset_disp_maps.data_vars
        assert np.all(dataset_disp_maps["confidence_measure"].values == 0)

    @pytest.fixture()
    def expected_value_with_one_peak(self, disps_row, disps_col):
        """Compute expected ambiguity value for one peak"""
        # return 1 - [1 / nbr_disparity]
        return 1 - (1 / float(len(disps_col) * len(disps_row)))

    @pytest.mark.parametrize("cost_volume_init_value", [0])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test_with_one_peak(
        self,
        cost_volume_confidence_instance,
        empty_dataset,
        left_datasets,
        cost_volume,
        dataset_disp_maps,
        expected_value_with_one_peak,
    ):
        """
        Test confidence_prediction method with monotonic surface
        i.e. cost_surface is filled only with one peak (value = 1) and the remaining elements to cost_volume_init_value
        Confidence is at its highest, i.e. 1
        """

        # For all points, there is only one peak where row = 0 and col = 0
        cost_volume["cost_volumes"].values[:, :, 1, 3] = 1

        _, dataset_disp_maps = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume,
            dataset_disp_maps=dataset_disp_maps,
        )

        assert "confidence_measure" in dataset_disp_maps.data_vars
        assert np.all(dataset_disp_maps["confidence_measure"].values == expected_value_with_one_peak)

    @pytest.mark.parametrize("cost_volume_init_value", [0])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    def test_with_multiple_peak(
        self,
        cost_volume_confidence_instance,
        empty_dataset,
        left_datasets,
        cost_volume,
        dataset_disp_maps,
        expected_value_with_one_peak,
    ):
        """
        Test confidence_prediction method with multiple peak
        i.e. cost_surface is filled with multiple max value and the remaining elements to cost_volume_init_value
        Here we are checking whether the result is the same regardless of whether the peak is in the rows or the columns
        """

        cost_volume_row = copy.deepcopy(cost_volume)
        cost_volume_column = copy.deepcopy(cost_volume)

        # Add a peak every 2 row disparities
        cost_volume_row["cost_volumes"].values[:, :, ::2, :] = 0.8
        # Add a peak every 2 column disparities
        cost_volume_column["cost_volumes"].values[:, :, :, ::2] = 0.8

        _, dataset_disp_maps_row = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume_row,
            dataset_disp_maps=dataset_disp_maps,
        )
        _, dataset_disp_maps_column = cost_volume_confidence_instance.confidence_prediction(
            left_image=left_datasets,
            right_image=empty_dataset,
            cost_volumes=cost_volume_column,
            dataset_disp_maps=dataset_disp_maps,
        )

        assert "confidence_measure" in dataset_disp_maps_row.data_vars
        assert "confidence_measure" in dataset_disp_maps_column.data_vars
        np.testing.assert_array_equal(
            dataset_disp_maps_row["confidence_measure"], dataset_disp_maps_column["confidence_measure"]
        )
        assert np.all(dataset_disp_maps_row["confidence_measure"].values < expected_value_with_one_peak)
