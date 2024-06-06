# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France.
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
"""Module with global test fixtures."""

from copy import deepcopy
import numpy as np
import xarray as xr
from rasterio import Affine
import pytest
from skimage.io import imsave

from pandora2d import Pandora2DMachine


@pytest.fixture()
def pandora2d_machine():
    """pandora2d_machine"""
    return Pandora2DMachine()


@pytest.fixture()
def pipeline_config(correct_pipeline):
    return deepcopy(correct_pipeline)


@pytest.fixture()
def make_empty_image(tmp_path):
    """Returns an empty image factory.

    temporary dir is the same accross calls in for the same test so if multiple images need to be created ensure to
    give a different name for each one.
    """

    def make(name="empty.tiff", shape=(450, 450)):
        """
        Make an empty image and return its path.

        :param name: name of the image
        :type name: str
        :param shape: shape of the image
        :type shape: Tuple[int, int]
        :return: image path
        :rtype: pathlib.Path
        """
        path = tmp_path / name
        imsave(path, np.empty(shape))
        return path

    return make


@pytest.fixture
def correct_input_cfg(left_img_path, right_img_path):
    return {
        "input": {
            "left": {
                "img": left_img_path,
                "nodata": -9999,
            },
            "right": {"img": right_img_path, "nodata": -9999},
            "col_disparity": [-2, 2],
            "row_disparity": [-2, 2],
        }
    }


@pytest.fixture
def false_input_path_image(right_img_path):
    return {
        "input": {
            "left": {
                "img": "./tests/data/lt.png",
                "nodata": "NaN",
            },
            "right": {
                "img": right_img_path,
            },
            "col_disparity": [-2, 2],
            "row_disparity": [-2, 2],
        }
    }


@pytest.fixture
def false_input_disp(left_img_path, right_img_path):
    return {
        "input": {
            "left": {
                "img": left_img_path,
            },
            "right": {
                "img": right_img_path,
            },
            "col_disparity": [7, 2],
            "row_disparity": [-2, 2],
        }
    }


@pytest.fixture(name="correct_pipeline")
def correct_pipeline_fixture():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "interpolation"},
        }
    }


@pytest.fixture
def false_pipeline_mc():
    return {
        "pipeline": {
            "disparity": {"disparity_method": "wta", "invalid_disparity": -99},
            "refinement": {"refinement_method": "interpolation"},
        }
    }


@pytest.fixture
def false_pipeline_disp():
    return {
        "pipeline": {
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5},
            "refinement": {"refinement_method": "interpolation"},
        }
    }


@pytest.fixture
def correct_roi_sensor():
    return {
        "ROI": {
            "col": {"first": 10, "last": 100},
            "row": {"first": 10, "last": 100},
        }
    }


@pytest.fixture
def false_roi_sensor_negative():
    return {
        "ROI": {
            "col": {"first": -10, "last": 100},
            "row": {"first": 10, "last": 100},
        }
    }


@pytest.fixture
def false_roi_sensor_first_superior_to_last():
    return {
        "ROI": {
            "col": {"first": 110, "last": 100},
            "row": {"first": 10, "last": 100},
        }
    }


@pytest.fixture()
def left_stereo_object():
    """
    Create left stereo object
    """

    data = np.array(([-9999, -9999, -9999], [1, 1, 1], [3, 4, 5]), dtype=np.float64)

    mask = np.array(([1, 1, 1], [0, 0, 0], [0, 0, 0]), dtype=np.int16)

    left = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "col_disparity_source": [0, 1],
        "row_disparity_source": [-1, 0],
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    return left


@pytest.fixture()
def right_stereo_object():
    """
    Create right stereo object
    """

    data = np.array(
        ([1, 1, 1], [3, 4, 5], [1, 1, 1]),
        dtype=np.float64,
    )
    mask = np.array(([0, 0, 0], [0, 0, 0], [0, 0, 0]), dtype=np.int16)
    right = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    return right


@pytest.fixture()
def stereo_object_with_args():
    """
    Create a stero object with some arguments
    """
    data = np.array(
        ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    left_arg = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    left_arg.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "col_disparity_source": [0, 1],
        "row_disparity_source": [-1, 0],
    }

    data = np.array(
        ([[1, 1, 1, 1, 1], [3, 4, 5, 6, 7], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        dtype=np.float64,
    )
    mask = np.array(
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16
    )
    right_arg = xr.Dataset(
        {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )
    right_arg.attrs = {
        "no_data_img": -9999,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
    }

    return left_arg, right_arg
