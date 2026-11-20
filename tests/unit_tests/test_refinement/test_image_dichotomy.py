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
"""
Test the refinement.dichotomy module for image resampling.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

import copy

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import shift as shift_image

from pandora2d import common, criteria, disparity, refinement
from pandora2d.matching_cost import MatchingCostRegistry


class TestImageResampling:
    """
    Test dichotomy refinement by resampling the right image.
    """

    @pytest.fixture()
    def image_size(self):
        """
        Image size
        """
        return 80

    @pytest.fixture()
    def left_image_dataset(self, image_size):
        """
        Left image dataset
        """

        rows = np.arange(image_size)
        cols = np.arange(image_size)
        left_data = np.random.default_rng(42).normal(size=(image_size, image_size)).astype(np.float32)

        # Row disparity range is [2, 7] and column disparity range is [-2, 3]
        row_disparity = np.stack((np.full(left_data.shape, 2), np.full(left_data.shape, 7)))
        col_disparity = np.stack((np.full(left_data.shape, -2), np.full(left_data.shape, 3)))
        return xr.Dataset(
            {
                "im": (["row", "col"], left_data),
                "row_disparity": (
                    ["band_disp", "row", "col"],
                    row_disparity,
                    {"no_data": None},
                ),
                "col_disparity": (
                    ["band_disp", "row", "col"],
                    col_disparity,
                    {"no_data": None},
                ),
            },
            coords={"row": rows, "col": cols, "band_disp": ["min", "max"]},
            attrs={
                "no_data_img": np.nan,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "row_disparity_source": [2, 7],
                "col_disparity_source": [-2, 3],
            },
        )

    @pytest.fixture()
    def right_image_dataset(self, left_image_dataset, gt_disparities):
        """
        Create a right image dataset translated by the known subpixel disparities.
        """
        right_data = shift_image(
            left_image_dataset["im"].data,
            shift=gt_disparities,
            order=1,
            mode="constant",
            cval=0,
        )
        right_dataset = left_image_dataset.copy(
            deep=True,
            data={
                "im": right_data,
                "row_disparity": -left_image_dataset["row_disparity"].data[::-1],
                "col_disparity": -left_image_dataset["col_disparity"].data[::-1],
            },
        )
        for axis in ("row", "col"):
            source = left_image_dataset.attrs[f"{axis}_disparity_source"]
            right_dataset.attrs[f"{axis}_disparity_source"] = [-source[1], -source[0]]
        return right_dataset

    @pytest.fixture()
    def cost_volumes(self, matching_cost_config, left_image_dataset, right_image_dataset):
        """
        Return cost volumes
        """
        matching_cost = MatchingCostRegistry.get(matching_cost_config["matching_cost_method"])(matching_cost_config)
        matching_cost.allocate(left_image_dataset, right_image_dataset, matching_cost_config)
        return matching_cost.compute_cost_volumes(left_image_dataset, right_image_dataset)

    @pytest.fixture()
    def disparity_maps(self, cost_volumes):
        """
        Return disparity maps
        """
        dataset_validity = criteria.get_validity_dataset(
            cost_volumes["criteria"],
            cost_volumes.attrs["row_disparity_source"],
            cost_volumes.attrs["col_disparity_source"],
        )
        disparity_maps = common.dataset_disp_maps(
            cost_volumes.coords,
            dataset_validity,
            {"invalid_disp": np.nan},
        )
        col_map, row_map, correlation_score = disparity.Disparity(
            {"disparity_method": "wta", "invalid_disparity": np.nan}
        ).compute_disp_maps(cost_volumes)
        common.complete_dataset_disp_maps(disparity_maps, row_map, col_map, correlation_score)
        return disparity_maps

    @pytest.fixture()
    def iterations(self):
        """
        Number of dichotomy iterations
        """
        return 1

    @pytest.fixture()
    def image_dichotomy_instance(self, iterations):
        """Create a dichotomy instance configured for image resampling."""
        return refinement.dichotomy_cpp.Dichotomy(
            {
                "refinement_method": "dichotomy",
                "iterations": iterations,
                "filter": {"method": "bicubic"},
                "resampling_type": "image",
            }
        )

    @staticmethod
    def compute_padding(window_size, cost_volumes, gt_disparities, image_dichotomy_instance):
        """
        Compute the padding to exclude image borders from the test
        because the borders can skew the results of the dichotomy.

        :param window_size: correlation window size
        :param cost_volumes: cost volumes containing the disparity ranges
        :param gt_disparities: row and column disparities used to translate the right image
        :param image_dichotomy_instance: dichotomy instance containing the interpolation filter
        :return: padding to exclude from each image border
        """
        disparity_range = (*cost_volumes.attrs["row_disparity_source"], *cost_volumes.attrs["col_disparity_source"])
        margins = image_dichotomy_instance.filter.margins

        # Half window size around the current point
        window_radius = window_size // 2
        # Maximum displacement in the row and column disparity ranges
        search_margin = int(np.ceil(max(abs(value) for value in disparity_range)))
        # Maximum translation of the right image,
        # rounded up to cover the added zeros by scipy.shift_image method
        translation_margin = int(np.ceil(max(abs(value) for value in gt_disparities)))
        # Margin of the interpolation filter
        interpolation_margin = max(margins.left, margins.right, margins.up, margins.down)
        # Add one pixel for the displacements tested by the dichotomy
        return window_radius + search_margin + translation_margin + interpolation_margin + 1

    # Tests using mutual information work starting at window_size = 11
    @pytest.mark.parametrize("min_ratio", [pytest.param(0.9, id="min ratio 90 percent")])
    @pytest.mark.parametrize("gt_disparities", [(4.5, 0.5), (5.5, -0.5)])
    @pytest.mark.parametrize("window_size", [11, 13])
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    def test_result_of_one_iteration(
        self,
        image_dichotomy_instance,
        left_image_dataset,
        right_image_dataset,
        cost_volumes,
        disparity_maps,
        gt_disparities,
        min_ratio,
        window_size,
    ):
        """
        Test result of one iteration of image resampling refinement method
        """

        copy_disp_map = copy.deepcopy(disparity_maps)

        result_disp_col, result_disp_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_image_dataset, right_image_dataset
        )

        gt_disp_row, gt_disp_col = gt_disparities

        # Compute the padding to exclude image borders from the test
        padding = self.compute_padding(window_size, cost_volumes, gt_disparities, image_dichotomy_instance)
        result_disp_col = result_disp_col[padding:-padding, padding:-padding]
        result_disp_row = result_disp_row[padding:-padding, padding:-padding]

        # Proportion of valid points equal to the expected disparity
        ratio_col = np.mean(result_disp_col == gt_disp_col)
        ratio_row = np.mean(result_disp_row == gt_disp_row)

        assert ratio_col >= min_ratio
        assert ratio_row >= min_ratio

    # Tests using mutual information work starting at window_size = 11
    @pytest.mark.parametrize("min_ratio", [pytest.param(0.5, id="min ratio 50 percent")])
    @pytest.mark.parametrize("iterations", [2])
    @pytest.mark.parametrize("gt_disparities", [(2.5, 0.25), (4.25, -0.75)])
    @pytest.mark.parametrize("window_size", [11, 13])
    @pytest.mark.parametrize("subpix", [2])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    def test_result_of_two_iterations(
        self,
        image_dichotomy_instance,
        left_image_dataset,
        right_image_dataset,
        cost_volumes,
        disparity_maps,
        gt_disparities,
        min_ratio,
        window_size,
    ):
        """
        Test result of two iterations of image resampling refinement method
        """

        copy_disp_map = copy.deepcopy(disparity_maps)

        result_disp_col, result_disp_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy_disp_map, left_image_dataset, right_image_dataset
        )

        gt_disp_row, gt_disp_col = gt_disparities
        # Compute the padding to exclude image borders from the test
        padding = self.compute_padding(window_size, cost_volumes, gt_disparities, image_dichotomy_instance)
        result_disp_col = result_disp_col[padding:-padding, padding:-padding]
        result_disp_row = result_disp_row[padding:-padding, padding:-padding]

        # Proportion of valid points equal to the expected disparity
        ratio_col = np.mean(result_disp_col == gt_disp_col)
        ratio_row = np.mean(result_disp_row == gt_disp_row)

        assert ratio_col >= min_ratio
        assert ratio_row >= min_ratio

    @pytest.mark.parametrize(
        ("gt_disparities", "tolerance", "min_ratio"),
        [
            pytest.param((4.125, 0.125), 0.0625, 0.75),
            pytest.param((4.0625, 0.0625), 0, 0.5),
        ],
    )
    # Tests using mutual information work starting at window_size = 11
    @pytest.mark.parametrize("iterations", [4])
    @pytest.mark.parametrize("window_size", [11, 13])
    @pytest.mark.parametrize("subpix", [4])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    def test_result_of_four_iterations(
        self,
        image_dichotomy_instance,
        left_image_dataset,
        right_image_dataset,
        cost_volumes,
        disparity_maps,
        gt_disparities,
        tolerance,
        min_ratio,
        window_size,
    ):
        """
        Test result of four iterations of image resampling refinement method
        """
        result_disp_col, result_disp_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )

        gt_disp_row, gt_disp_col = gt_disparities
        padding = self.compute_padding(window_size, cost_volumes, gt_disparities, image_dichotomy_instance)
        result_disp_col = result_disp_col[padding:-padding, padding:-padding]
        result_disp_row = result_disp_row[padding:-padding, padding:-padding]

        # A tolerance of one sixteenth is used for eighth-pixel disparities.
        ratio_col = np.mean(np.abs(result_disp_col - gt_disp_col) <= tolerance)
        ratio_row = np.mean(np.abs(result_disp_row - gt_disp_row) <= tolerance)

        assert ratio_col >= min_ratio
        assert ratio_row >= min_ratio

    @pytest.mark.parametrize("gt_disparities", [(4.5, 0.5)])
    @pytest.mark.parametrize("window_size", [11])
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    def test_with_nans_in_disp_map(
        self, image_dichotomy_instance, left_image_dataset, right_image_dataset, cost_volumes, disparity_maps
    ):
        """
        NaN disparities remain invalid while other points are refined
        """
        disparity_maps["row_map"].data[30, 32] = np.nan
        disparity_maps["col_map"].data[30, 32] = np.nan
        disparity_maps["validity"].loc[{"criteria": "partial_validity_mask"}].data[30, 32] = 1

        result_col, result_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )

        # Check that point (30,32) is still NaN in refined disparity maps
        assert np.isnan(result_row[30, 32])
        assert np.isnan(result_col[30, 32])
        # Check that neighbour of (30,32) is correctly refined
        assert result_row[30, 31] == 4.5
        assert result_col[30, 31] == 0.5
        assert result_row[30, 31] != disparity_maps["row_map"].data[30, 31]
        assert result_col[30, 31] != disparity_maps["col_map"].data[30, 31]

    @pytest.fixture()
    def right_mask(self, right_image_dataset, mask_type):
        """
        Right mask with invalid pixels around the point (30, 31)
        """
        mask = xr.zeros_like(right_image_dataset["im"], dtype=np.uint8)
        no_data_mask = right_image_dataset.attrs["no_data_mask"]
        mask_value = no_data_mask
        if mask_type == "invalid":
            mask_value = right_image_dataset.attrs["valid_pixels"] + no_data_mask + 1
        mask.data[34:37, 30:33] = mask_value
        return mask

    @pytest.mark.parametrize(
        "subpix, iterations, gt_disparities",
        [
            pytest.param(1, 1, (4.5, 0.5), id="half pixel"),
            pytest.param(2, 2, (4.25, -0.75), id="quarter pixel"),
        ],
    )
    @pytest.mark.parametrize("window_size", [11])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    @pytest.mark.parametrize(
        "mask_type",
        [
            pytest.param("no_data", id="no data points in mask"),
            pytest.param("invalid", id="invalid points in mask"),
        ],
    )
    def test_with_invalid_pixel_in_right_image(
        self,
        image_dichotomy_instance,
        left_image_dataset,
        right_image_dataset,
        cost_volumes,
        disparity_maps,
        right_mask,
    ):
        """
        Test that invalid pixels in the right mask prevent refinement of point (30, 31)
        """
        # Point (30, 31) is refined when the right image is valid
        valid_col, valid_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )
        assert (
            valid_row[30, 31] != disparity_maps["row_map"].data[30, 31]
            or valid_col[30, 31] != disparity_maps["col_map"].data[30, 31]
        )

        right_image_dataset["msk"] = right_mask

        result_col, result_row, result_score = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )

        # All candidates at (30, 31) have an invalid interpolation support
        assert result_row[30, 31] == disparity_maps["row_map"].data[30, 31]
        assert result_col[30, 31] == disparity_maps["col_map"].data[30, 31]
        initial_score = refinement.dichotomy_cpp.create_cost_values_map(cost_volumes, disparity_maps)
        assert result_score[30, 31] == initial_score[30, 31]
        # Point (40, 40) remains outside the masked area and is refined
        assert (
            result_row[40, 40] != disparity_maps["row_map"].data[40, 40]
            or result_col[40, 40] != disparity_maps["col_map"].data[40, 40]
        )

    @pytest.mark.parametrize("gt_disparities", [(4.5, 0.5)])
    @pytest.mark.parametrize("window_size", [11])
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("matching_cost_method", ["zncc"])
    def test_with_invalid_pixel_in_some_interpolation_supports(
        self, image_dichotomy_instance, left_image_dataset, right_image_dataset, cost_volumes, disparity_maps
    ):
        """
        A point can be refined if a candidate has a valid interpolation support
        """
        point_near_invalid_pixel = (30, 31)
        # Check the result for the target point with a valid right image
        valid_disp_col, valid_disp_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )
        assert valid_disp_row[point_near_invalid_pixel] == 4.5
        assert valid_disp_col[point_near_invalid_pixel] == 0.5

        # Add an invalid pixel in the best candidate's support but not in every candidate's support
        right_image_dataset["msk"] = xr.zeros_like(right_image_dataset["im"], dtype=np.uint8)
        right_image_dataset["msk"].data[28, 38] = right_image_dataset.attrs["no_data_mask"]
        result_col, result_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )

        # The best candidate is skipped but another valid candidate still refines the column disparity
        assert result_row[point_near_invalid_pixel] == disparity_maps["row_map"].data[point_near_invalid_pixel]
        assert result_col[point_near_invalid_pixel] == 0.5
        assert result_col[point_near_invalid_pixel] != disparity_maps["col_map"].data[point_near_invalid_pixel]

    @pytest.mark.parametrize("gt_disparities", [(4.5, 0.5)])
    @pytest.mark.parametrize("window_size", [11])
    @pytest.mark.parametrize("subpix", [1])
    @pytest.mark.parametrize("matching_cost_method", ["mutual_information", "zncc"])
    @pytest.mark.parametrize("invalid_disparity", [-9999])
    def test_with_invalid_values_in_disp_map(
        self,
        image_dichotomy_instance,
        left_image_dataset,
        right_image_dataset,
        cost_volumes,
        disparity_maps,
        invalid_disparity,
    ):
        """
        Invalid row or column disparities are preserved while other points are refined
        """
        disparity_maps.attrs["invalid_disp"] = invalid_disparity
        for key in ("row_map", "col_map"):
            disparity_maps[key].data[np.isnan(disparity_maps[key].data)] = invalid_disparity
        disparity_maps["row_map"].data[30, 32] = invalid_disparity
        disparity_maps["col_map"].data[30, 33] = invalid_disparity
        disparity_maps["validity"].loc[{"criteria": "partial_validity_mask"}].data[30, 32:34] = 1

        result_col, result_row, _ = image_dichotomy_instance.refinement_method(
            cost_volumes, copy.deepcopy(disparity_maps), left_image_dataset, right_image_dataset
        )

        # Check that points (30,32) and (30,33) are still invalid in refined disparity maps
        assert result_row[30, 32] == invalid_disparity
        assert result_col[30, 32] == disparity_maps["col_map"].data[30, 32]
        assert result_col[30, 33] == invalid_disparity
        assert result_row[30, 33] == disparity_maps["row_map"].data[30, 33]
        # Check that neighbour of (30,31) is correctly refined
        assert result_row[30, 31] == 4.5
        assert result_col[30, 31] == 0.5
        assert result_row[30, 31] != disparity_maps["row_map"].data[30, 31]
        assert result_col[30, 31] != disparity_maps["col_map"].data[30, 31]
