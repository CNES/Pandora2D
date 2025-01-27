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
import json

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import xarray as xr
import rasterio
from pytest_mock import MockerFixture
from skimage.io import imsave
from rasterio import Affine
from pandora2d import common, run
from pandora2d.check_configuration import check_conf
from pandora2d.img_tools import create_datasets_from_inputs
from pandora2d import matching_cost, disparity, refinement
from pandora2d.state_machine import Pandora2DMachine
from pandora2d.constants import Criteria


@pytest.fixture(scope="class")
def save_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("save_folder")


class TestRegistry:
    """Test Registry behavior."""

    def test_registering(self):
        """When registring a class with name, we should be able to get it with get."""
        registry = common.Registry()  # type: ignore[var-annotated]

        @registry.add("example")
        class Example:  # pylint: disable=too-few-public-methods
            pass

        assert registry.get("example") is Example

    def test_is_generic(self):
        """We should be able to specify the returned base class type."""

        class BaseExample:  # pylint: disable=too-few-public-methods
            pass

        common.Registry[BaseExample]()

    def test_get_unregistered_raise_error(self):
        """Without default, we raise a KeyError if name is not registered."""
        registry = common.Registry()  # type: ignore[var-annotated]

        with pytest.raises(KeyError, match=r"No class registered with name `example`\."):
            assert registry.get("example")

    def test_can_have_default(self):
        """When a default is given it should be return for unregistered names."""

        class Example:  # pylint: disable=too-few-public-methods
            pass

        registry = common.Registry(default=Example)

        assert registry.get("unregistered") is Example


class TestSaveConfig:
    """Test save_config behavior."""

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("my_output", id="simple"),
            pytest.param("subdir/my_output", id="with subdir"),
        ],
    )
    def test_parents_do_not_exist(self, tmp_path, correct_input_cfg, correct_pipeline_without_refinement, path):
        """Parents do not exist."""
        output_path = tmp_path / path
        config = {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            "output": {"path": str(output_path)},
        }

        common.save_config(config)

        assert (output_path / "config.json").is_file()

    def test_parents_exist(self, tmp_path, correct_input_cfg, correct_pipeline_without_refinement):
        """Should not fail if parents exist."""
        output_path = tmp_path / "out"
        output_path.mkdir()
        config = {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            "output": {"path": str(output_path)},
        }

        common.save_config(config)

        assert (output_path / "config.json").is_file()

    def test_overwrite(self, tmp_path, correct_input_cfg, correct_pipeline_without_refinement):
        """Overwrite file in exists."""
        existing_file = tmp_path / "config.json"
        existing_file.write_text("not json loadable")

        config = {
            **correct_input_cfg,
            **correct_pipeline_without_refinement,
            "output": {"path": str(tmp_path)},
        }

        common.save_config(config)

        with existing_file.open() as fd:
            assert json.load(fd)


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "crs": "EPSG:32632",
            "transform": Affine(25.94, 0.00, -5278429.43, 0.00, -25.94, 14278941.03),
        },
        {"crs": None, "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)},
    ],
    scope="class",
)
class TestSaveDisparityMaps:
    """Test save_disparity_maps method"""

    @pytest.fixture(scope="class")
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

        dataset = xr.Dataset(
            {
                "kill_map": xr.DataArray(row, dims=dims, coords=coords),
                "power_map": xr.DataArray(col, dims=dims, coords=coords),
                "super_score": xr.DataArray(score, dims=dims, coords=coords),
            },
            attrs=attributes,
        )

        return dataset

    @pytest.fixture(scope="class")
    def fake_report_data(self):
        return {"answer": 42}

    @pytest.fixture(scope="class")
    def save_disparity_maps(
        self,
        save_folder,
        class_scoped_correct_input_cfg,
        create_test_dataset,
        fake_report_data,
        class_mocker: MockerFixture,
    ):
        """Fixture that saves disparity_map and return its path."""
        class_mocker.patch("pandora2d.common.reporting.report_disparities", return_value=fake_report_data)
        output_dir = save_folder / "output_dataset"
        common.save_disparity_maps(
            create_test_dataset,
            {**class_scoped_correct_input_cfg, "output": {"path": str(output_dir)}},
        )
        return output_dir

    @pytest.mark.parametrize(
        "file_name",
        ["power_map.tif", "kill_map.tif", "super_score.tif"],
    )
    def test_save_disparities(self, save_disparity_maps, attributes, file_name):
        """
        Function for testing the save_disparities function
        """

        file = save_disparity_maps / file_name

        assert save_disparity_maps.exists()
        assert file.exists()

        data = rasterio.open(file)

        assert data.crs == attributes["crs"]
        assert data.transform == attributes["transform"]

    def test_save_report(self, save_disparity_maps, fake_report_data):
        """Check report is generated by save_disparity_maps."""
        file = save_disparity_maps / "report.json"
        assert save_disparity_maps.exists()

        with file.open() as fd:
            result = json.load(fd)

        assert result == {"statistics": {"disparity": fake_report_data}}


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

        matching_cost_matcher = matching_cost.PandoraMatchingCostMethods(cfg["pipeline"]["matching_cost"])

        matching_cost_matcher.allocate(
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
    correct_input_cfg.update({"output": {"path": "tatoo"}})

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
            np.full((5, 4, 6, 7), init_value),
            coords={
                "row": range_row,
                "col": range_col,
                "disp_row": disp_range_row,
                "disp_col": disp_range_col,
            },
            dims=["row", "col", "disp_row", "disp_col"],
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

        expected_value = array.data[::2, :, :min_disp_index, :]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, :, min_disp_index:, :]

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

        expected_value = array.data[:, ::2, :, :min_disp_index]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, :, min_disp_index:]

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

        expected_value = array.data[::2, :, (max_disp_index + 1) :, :]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, :, : (max_disp_index + 1), :]

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

        expected_value = array.data[:, ::2, :, (max_disp_index + 1) :]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, :, : (max_disp_index + 1)]

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

        expected_below_min = array.data[::2, :, :min_disp_index, :]
        expected_above_max = array.data[::2, :, (max_disp_index + 1) :, :]
        expected_zeros_on_odd_lines = array.data[1::2, ...]
        expected_zeros_on_even_lines = array.data[::2, :, min_disp_index : (max_disp_index + 1), :]

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

        expected_below_min = array.data[:, ::2, :, :min_disp_index]
        expected_above_max = array.data[:, ::2, :, (max_disp_index + 1) :]
        expected_zeros_on_odd_columns = array.data[:, 1::2, ...]
        expected_zeros_on_even_columns = array.data[:, ::2, :, min_disp_index : (max_disp_index + 1)]

        assert np.all(expected_below_min == value)
        assert np.all(expected_above_max == value)
        assert np.all(expected_zeros_on_odd_columns == init_value)
        assert np.all(expected_zeros_on_even_columns == init_value)
