#!/usr/bin/env python
# coding: utf8
#
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
Test common
"""

# pylint: disable=redefined-outer-name
import os

import numpy as np
import pytest
import xarray as xr
import rasterio
from skimage.io import imsave
from rasterio import Affine
from pandora2d import common, run
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d import matching_cost, disparity, refinement
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.constants import Criteria


class TestSaveDataset:
    """Test save_dataset method"""

    @pytest.fixture
    def create_test_dataset(self, attributes):
        """
        Create a test dataset
        """
        row, col, score = np.full((2, 2), 1), np.full((2, 2), 1), np.full((2, 2), 1)

        coords = {
            "row": np.arange(row.shape[0]),
            "col": np.arange(col.shape[1]),
        }

        dims = ("row", "col")

        dataarray_row = xr.DataArray(row, dims=dims, coords=coords)
        dataarray_col = xr.DataArray(col, dims=dims, coords=coords)
        dataarray_score = xr.DataArray(score, dims=dims, coords=coords)

        dataset = xr.Dataset(
            {"row_map": dataarray_row, "col_map": dataarray_col, "correlation_score": dataarray_score},
            attrs=attributes,
        )

        return dataset

    @pytest.mark.parametrize(
        "attributes",
        [
            {"crs": "EPSG:32632", "transform": Affine(25.94, 0.00, -5278429.43, 0.00, -25.94, 14278941.03)},
            {"crs": None, "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)},
        ],
    )
    def test_save_dataset(self, create_test_dataset, correct_input_cfg, attributes):
        """
        Function for testing the dataset_save function
        """

        common.save_dataset(create_test_dataset, correct_input_cfg, "./tests/res_test/")
        assert os.path.exists("./tests/res_test/")

        # Test columns disparity map
        assert os.path.exists("./tests/res_test/columns_disparity.tif")
        columns_disparity = rasterio.open("./tests/res_test/columns_disparity.tif")
        assert columns_disparity.crs == attributes["crs"]
        assert columns_disparity.transform == attributes["transform"]

        # Test row disparity map
        assert os.path.exists("./tests/res_test/row_disparity.tif")
        row_disparity = rasterio.open("./tests/res_test/row_disparity.tif")
        assert row_disparity.crs == attributes["crs"]
        assert row_disparity.transform == attributes["transform"]

        # Test correlation score map
        assert os.path.exists("./tests/res_test/correlation_score.tif")
        correlation_score = rasterio.open("./tests/res_test/correlation_score.tif")
        assert correlation_score.crs == attributes["crs"]
        assert correlation_score.transform == attributes["transform"]

        os.remove("./tests/res_test/columns_disparity.tif")
        os.remove("./tests/res_test/row_disparity.tif")
        os.remove("./tests/res_test/correlation_score.tif")
        os.rmdir("./tests/res_test")


def create_dataset_coords(data_row, data_col, data_score, row, col):
    """
    Create xr.Dataset with data_row and data_col as data variables and row and col as coordinates
    """

    data_variables = {
        "row_map": (("row", "col"), data_row),
        "col_map": (("row", "col"), data_col),
        "correlation_score": (("row", "col"), data_score),
    }

    coords = {"row": row, "col": col}

    dataset = xr.Dataset(data_variables, coords)

    return dataset


class TestDatasetDispMaps:
    """Test dataset_disp_maps method"""

    @pytest.fixture()
    def left_image(self, tmp_path):
        """
        Create a fake image to test dataset_disp_maps method
        """
        image_path = tmp_path / "left_img.tif"
        data = np.full((10, 10), 1, dtype=np.uint8)
        imsave(image_path, data)

        return image_path

    @pytest.fixture()
    def right_image(self, tmp_path):
        """
        Create a fake image to test dataset_disp_maps method
        """
        image_path = tmp_path / "right_img.tif"
        data = np.full((10, 10), 1, dtype=np.uint8)
        imsave(image_path, data)

        return image_path

    @pytest.mark.parametrize(
        ["row", "col"],
        [
            pytest.param(
                np.arange(10),
                np.arange(10),
                id="Classic case",
            ),
            pytest.param(
                np.arange(10, 20),
                np.arange(20, 30),
                id="ROI case",
            ),
            pytest.param(
                np.arange(2, 12),
                np.arange(2, 12, 2),
                id="Step in col",
            ),
            pytest.param(
                np.arange(2, 12, 2),
                np.arange(2, 12, 2),
                id="Step in row",
            ),
        ],
    )
    def test_dataset_disp_maps(self, row, col):
        """
        Test for dataset_disp_maps method
        """

        dataset_test = create_dataset_coords(
            np.full((len(row), len(col)), 1),
            np.full((len(row), len(col)), 1),
            np.full((len(row), len(col)), 1),
            row,
            col,
        )

        # create dataset with dataset_disp_maps function
        disparity_maps = common.dataset_disp_maps(
            dataset_test.row_map,
            dataset_test.col_map,
            dataset_test.coords,
            np.full((len(row), len(col)), 1),
            {"invalid_disp": -9999},
        )

        assert disparity_maps.equals(dataset_test)

    @pytest.mark.parametrize(
        ["coord_value", "coord", "string_match"],
        [
            pytest.param(
                np.arange(10),
                "row",
                "The col coordinate does not exist",
                id="No col coordinates",
            ),
            pytest.param(
                np.arange(10),
                "col",
                "The row coordinate does not exist",
                id="No row coordinates",
            ),
        ],
    )
    def test_dataset_disp_maps_fails_with_missing_coords(self, coord_value, coord, string_match):
        """
        Test that dataset_disp_maps method fails when one of the coordinates is missing
        """

        # create a dataset with only one of the two required coordinates
        data_variables = {
            "row_map": ((coord), np.full((len(coord_value)), 1)),
            "col_map": ((coord), np.full((len(coord_value)), 1)),
        }

        coords = {coord: coord_value}

        dataset_test = xr.Dataset(data_variables, coords)

        # create dataset with dataset_disp_maps function
        with pytest.raises(ValueError, match=string_match):
            common.dataset_disp_maps(
                dataset_test.row_map,
                dataset_test.col_map,
                dataset_test.coords,
                np.full((len(coord_value)), 1),
                {"invalid_disp": -9999},
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        ["roi", "step"],
        [
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                [1, 1],
                id="ROI in image",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                [2, 1],
                id="ROI in image and step=2 for rows",
            ),
            pytest.param(
                None,
                [1, 1],
                id="No ROI",
            ),
        ],
    )
    def test_dataset_disp_maps_with_pipeline_computation(self, roi, step, left_image, right_image):
        """
        Test for dataset_disp_maps method after computation of disparity maps and refinement step
        """

        # input configuration
        input_cfg = {
            "left": {
                "img": left_image,
                "nodata": -9999,
            },
            "right": {
                "img": right_image,
                "nodata": -9999,
            },
            "col_disparity": {"init": 2, "range": 2},
            "row_disparity": {"init": 1, "range": 2},
        }

        img_left, img_right = create_datasets_from_inputs(input_cfg, roi=roi)

        if roi is not None:
            cfg = {
                "input": input_cfg,
                "pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3, "step": step}},
                "ROI": roi,
            }
        else:
            cfg = {
                "input": input_cfg,
                "pipeline": {"matching_cost": {"matching_cost_method": "zncc", "window_size": 3, "step": step}},
            }

        matching_cost_matcher = matching_cost.MatchingCost(cfg["pipeline"]["matching_cost"])

        matching_cost_matcher.allocate_cost_volume_pandora(
            img_left=img_left,
            img_right=img_right,
            cfg=cfg,
        )

        # compute cost volumes
        cvs = matching_cost_matcher.compute_cost_volumes(img_left=img_left, img_right=img_right)

        cfg_disp = {"disparity_method": "wta", "invalid_disparity": -9999}
        disparity_matcher = disparity.Disparity(cfg_disp)
        # compute disparity maps
        delta_row, delta_col, correlation_score = disparity_matcher.compute_disp_maps(cvs)

        # create dataset with dataset_disp_maps function
        disparity_maps = common.dataset_disp_maps(
            delta_row, delta_col, cvs.coords, correlation_score, {"invalid_disp": -9999}
        )

        interpolation = refinement.AbstractRefinement({"refinement_method": "interpolation"})  # type: ignore[abstract]
        # compute refined disparity maps
        delta_x, delta_y, correlation_score = interpolation.refinement_method(cvs, disparity_maps, img_left, img_right)

        # create dataset with dataset_disp_maps function
        disparity_maps["row_map"].data = delta_y
        disparity_maps["col_map"].data = delta_x
        disparity_maps["correlation_score"].data = correlation_score

        # create ground truth with create_dataset_coords method
        dataset_ground_truth = create_dataset_coords(
            delta_x, delta_y, correlation_score, disparity_maps.row, disparity_maps.col
        )

        assert disparity_maps.equals(dataset_ground_truth)


def test_disparity_map_output_georef(correct_pipeline, correct_input_cfg):
    """
    Test outputs georef with crs and transform
    """
    img_left, img_right = create_datasets_from_inputs(input_config=correct_input_cfg["input"])

    # Stock crs and transform information from input
    img_left.attrs["crs"] = "EPSG:32632"
    img_left.attrs["transform"] = Affine(25.94, 0.00, -5278429.43, 0.00, -25.94, 14278941.03)

    pandora2d_machine = Pandora2DMachine()
    # Delete refinement to fastest result
    del correct_pipeline["pipeline"]["refinement"]

    correct_input_cfg.update(correct_pipeline)

    checked_cfg = check_conf(correct_input_cfg, pandora2d_machine)

    dataset, _ = run(pandora2d_machine, img_left, img_right, checked_cfg)

    assert "EPSG:32632" == dataset.attrs["crs"]
    assert Affine(25.94, 0.00, -5278429.43, 0.00, -25.94, 14278941.03) == dataset.attrs["transform"]


class TestSetOutOfDisparity:
    """Test effect of disparity grids."""

    @pytest.fixture()
    def disp_coords(self):
        return "disp_row"

    @pytest.fixture()
    def init_value(self):
        return 0.0

    @pytest.fixture()
    def range_col(self):
        return np.arange(4)

    @pytest.fixture()
    def range_row(self):
        return np.arange(5)

    @pytest.fixture()
    def disp_range_col(self):
        return np.arange(2, 2 + 7)

    @pytest.fixture()
    def disp_range_row(self):
        return np.arange(-5, -5 + 6)

    @pytest.fixture()
    def dataset(self, range_row, range_col, disp_range_col, disp_range_row, init_value, disp_coords):
        """make a xarray dataset and disparity grids"""
        xarray = xr.DataArray(
            np.full((5, 4, 7, 6), init_value),
            coords={
                "row": range_row,
                "col": range_col,
                "disp_col": disp_range_col,
                "disp_row": disp_range_row,
            },
            dims=["row", "col", "disp_col", "disp_row"],
        )

        xarray.attrs = {"col_disparity_source": [2, 8], "row_disparity_source": [-5, 0]}
        min_disp_grid = np.full((xarray.sizes["row"], xarray.sizes["col"]), xarray.coords[disp_coords].data[0])
        max_disp_grid = np.full((xarray.sizes["row"], xarray.sizes["col"]), xarray.coords[disp_coords].data[-1])
        return xarray, min_disp_grid, max_disp_grid

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, np.nan],
            [0.0, 1],
            [0.0, -1],
            [0.0, np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    def test_homogeneous_row_grids(self, dataset, value):
        """With grids set to extreme disparities, cost_volumes should be left untouched."""
        # As set_out_of_row_disparity_range_to_other_value modify cost_volumes in place we do a copy to be able
        # to make the comparison later.
        array, min_disp_grid, max_disp_grid = dataset
        make_array_copy = array.copy(deep=True)
        common.set_out_of_row_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        xr.testing.assert_equal(array, make_array_copy)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, np.nan],
            [0.0, 10],
            [0.0, -10],
            [0.0, np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_homogeneous_col_grids(self, dataset, value):
        """With grids set to extreme disparities, cost_volumes should be left untouched."""
        # As set_out_of_col_disparity_range_to_other_value modify cost_volumes in place we do a copy to be able
        # to make the comparison later.
        array, min_disp_grid, max_disp_grid = dataset
        make_array_copy = array.copy(deep=True)
        common.set_out_of_col_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        xr.testing.assert_equal(array, make_array_copy)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    def test_variable_min_row(self, dataset, value, disp_coords, init_value):
        """Check special value below min disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[::2] = array.coords[disp_coords].data[min_disp_index]

        common.set_out_of_row_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_value = array.data[::2, ..., :min_disp_index]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, ..., min_disp_index:]

        assert np.all(expected_value == value)
        assert np.all(expected_zeros_on_odd_lines == init_value)
        assert np.all(expected_zeros_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_min_col(self, dataset, value, disp_coords, init_value):
        """Check special value below min disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[:, ::2] = array.coords[disp_coords].data[min_disp_index]

        common.set_out_of_col_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_value = array.data[:, ::2, :min_disp_index, ...]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, min_disp_index:, ...]

        assert np.all(expected_value == value)
        assert np.all(expected_zeros_on_odd_columns == init_value)
        assert np.all(expected_zeros_on_even_columns == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    def test_variable_max_row(self, dataset, value, disp_coords, init_value):
        """Check special value above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        max_disp_index = 1
        max_disp_grid[::2] = array.coords[disp_coords].data[max_disp_index]

        common.set_out_of_row_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_value = array.data[::2, ..., (max_disp_index + 1) :]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, ..., : (max_disp_index + 1)]

        assert np.all(expected_value == value)
        assert np.all(expected_zeros_on_odd_lines == init_value)
        assert np.all(expected_zeros_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_max_col(self, dataset, value, disp_coords, init_value):
        """Check special value above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        max_disp_index = 1
        max_disp_grid[:, ::2] = array.coords[disp_coords].data[max_disp_index]

        common.set_out_of_col_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_value = array.data[:, ::2, (max_disp_index + 1) :, ...]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, : (max_disp_index + 1), ...]

        assert np.all(expected_value == value)
        assert np.all(expected_zeros_on_odd_columns == init_value)
        assert np.all(expected_zeros_on_even_columns == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    def test_variable_min_and_max_row(self, dataset, value, disp_coords, init_value):
        """Check special value below min and above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[::2] = array.coords[disp_coords].data[min_disp_index]
        max_disp_index = 2
        max_disp_grid[::2] = array.coords[disp_coords].data[max_disp_index]

        common.set_out_of_row_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_below_min = array.data[::2, ..., :min_disp_index]
        expected_above_max = array.data[::2, ..., (max_disp_index + 1) :]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, ..., min_disp_index : (max_disp_index + 1)]

        assert np.all(expected_below_min == value)
        assert np.all(expected_above_max == value)
        assert np.all(expected_zeros_on_odd_lines == init_value)
        assert np.all(expected_zeros_on_even_lines == init_value)

    @pytest.mark.parametrize(
        ["init_value", "value"],
        [
            [0.0, 0.0],
            [0.0, -1],
            [0.0, np.inf],
            [0.0, -np.inf],
            [Criteria.VALID, Criteria.PANDORA2D_MSK_PIXEL_DISPARITY_UNPROCESSED],
        ],
    )
    @pytest.mark.parametrize("disp_coords", ["disp_col"])
    def test_variable_min_and_max_col(self, dataset, value, disp_coords, init_value):
        """Check special value below min and above max disparities."""
        array, min_disp_grid, max_disp_grid = dataset
        min_disp_index = 1
        min_disp_grid[:, ::2] = array.coords[disp_coords].data[min_disp_index]
        max_disp_index = 2
        max_disp_grid[:, ::2] = array.coords[disp_coords].data[max_disp_index]

        common.set_out_of_col_disparity_range_to_other_value(array, min_disp_grid, max_disp_grid, value)

        expected_below_min = array.data[:, ::2, :min_disp_index, ...]
        expected_above_max = array.data[:, ::2, (max_disp_index + 1) :, ...]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, min_disp_index : (max_disp_index + 1), ...]

        assert np.all(expected_below_min == value)
        assert np.all(expected_above_max == value)
        assert np.all(expected_zeros_on_odd_columns == init_value)
        assert np.all(expected_zeros_on_even_columns == init_value)
