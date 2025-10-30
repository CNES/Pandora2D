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

"""
Test multiscale check_conf method
"""

# pylint: disable=redefined-outer-name

import json
import pytest
import numpy as np
import rasterio

from json_checker import DictCheckerError
from pandora2d import multiscale


@pytest.fixture
def tmp_json_file(tmp_path):
    """
    Create a temporary json file
    """
    data = {"test_key": "test_value"}

    file_path = tmp_path / "data.json"

    with file_path.open("w", encoding="utf-8") as fd:
        json.dump(data, fd, indent=2)

    return file_path


@pytest.fixture()
def fake_json_file():
    """
    Create a path to a fake json file
    """

    return "path/fake.json"


@pytest.fixture()
def tmp_correct_repository(tmp_path_factory):
    """
    Create a temporary repository with two valid tif files (can be opened by rasterio).
    """
    tmp_path = tmp_path_factory.mktemp("correct_repository")
    data = np.zeros((10, 10))

    for i in range(2):
        path = tmp_path / f"image_{i}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype=data.dtype) as dst:
            dst.write(data, 1)

    return tmp_path


@pytest.fixture()
def tmp_repository_without_tif(tmp_path_factory):
    """
    Create a temporary repository without tif file.
    """

    tmp_path = tmp_path_factory.mktemp("repository_without_tif")
    (tmp_path / "readme.txt").write_text("txt file")
    (tmp_path / "data.csv").write_text("csv file")

    return tmp_path


@pytest.fixture()
def tmp_repository_with_unreadable_tif(tmp_path_factory):
    """
    Create a temporary repository with a corrupted tif file.
    """

    tmp_path = tmp_path_factory.mktemp("repository_with_wrong_tif")
    wrong_tif_path = tmp_path / "corrupted.tif"
    wrong_tif_path.write_text("Corrupted tif file")

    return tmp_path


@pytest.fixture()
def tmp_repository_with_three_files(tmp_path_factory):
    """
    Create a temporary repository with three valid tif files (can be opened by rasterio).
    """
    tmp_path = tmp_path_factory.mktemp("repository_with_three_files")
    data = np.zeros((10, 10))

    for i in range(3):
        path = tmp_path / f"image_{i}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype=data.dtype) as dst:
            dst.write(data, 1)

    return tmp_path


@pytest.fixture()
def correct_multiscale_config(tmp_correct_repository, tmp_json_file):
    """
    Correct multiscale configuration
    """
    return {
        "multiscale": {
            "left": {"pyramid": str(tmp_correct_repository)},
            "right": {"pyramid": str(tmp_correct_repository)},
            "model": {"type": "pol", "degree": 2},
            "scale_factors": [1, 2],
            "output": "output_test",
        },
        "pandora2d": str(tmp_json_file),
    }


@pytest.fixture()
def incorrect_multiscale_config(request):
    """
    Incorrect multiscale configuration
    """

    return {
        "multiscale": {
            "left": {"pyramid": str(request.getfixturevalue(request.param["left_path"]))},
            "right": {"pyramid": str(request.getfixturevalue(request.param["right_path"]))},
            "model": {"type": request.param["model_type"], "degree": request.param["model_degree"]},
            "scale_factors": request.param["scale_factors"],
            "output": request.param["output"],
        },
        "pandora2d": str(request.getfixturevalue(request.param["json_file"])),
    }


def test_multiscale_check_conf(correct_multiscale_config):
    """
    Test check_conf method for multiscale configuration
    """

    multiscale.check_configuration.check_conf(correct_multiscale_config)


def test_fails_if_multiscale_section_is_missing():
    """
    Test if multiscale section is missing in the configuration file
    """
    with pytest.raises(KeyError, match="multiscale key is missing"):
        multiscale.check_configuration.check_conf({})


def test_fails_if_pandora2d_section_is_missing(correct_multiscale_config):
    """
    Test if multiscale section is missing in the configuration file
    """

    del correct_multiscale_config["pandora2d"]

    with pytest.raises(KeyError, match="pandora2d key is missing"):
        multiscale.check_configuration.check_conf(correct_multiscale_config)


@pytest.mark.parametrize(
    ["incorrect_multiscale_config"],
    [
        pytest.param(
            {
                "left_path": "tmp_repository_without_tif",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Repository without tif file",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_repository_with_unreadable_tif",
                "model_type": "pol",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Repository with corrupted tif file",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2.2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Float model degree",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "wrong_model",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Wrong model type",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": 2,
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Wrong output type",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": 2,
                "json_file": "fake_json_file",
                "scale_factors": [1, 2],
            },
            id="Wrong pandora2d json file",
        ),
    ],
    indirect=["incorrect_multiscale_config"],
)
def test_fails_multiscale_check_conf(incorrect_multiscale_config):
    """
    Test that multiscale check_conf method fails when using wrong arguments
    """

    with pytest.raises(DictCheckerError):
        multiscale.check_configuration.check_conf(incorrect_multiscale_config)


@pytest.mark.parametrize(
    ["left_pyramid", "right_pyramid"],
    [
        pytest.param(True, False, id="Remove left pyramid"),
        pytest.param(False, True, id="Remove right pyramid"),
        pytest.param(True, True, id="Remove left pyramid & right pyramid"),
    ],
)
def test_fails_with_missing_keys(correct_multiscale_config, left_pyramid, right_pyramid):
    """
    Test that multiscale check_conf method fails when pyramid keys are missing
    """

    if left_pyramid:
        del correct_multiscale_config["multiscale"]["left"]["pyramid"]
    if right_pyramid:
        del correct_multiscale_config["multiscale"]["right"]["pyramid"]

    with pytest.raises(DictCheckerError):
        multiscale.check_configuration.check_conf(correct_multiscale_config)


def test_default_values(correct_multiscale_config):
    """
    Test that default values are correctly added to multiscale config
    """
    del correct_multiscale_config["multiscale"]["model"]

    result = multiscale.check_configuration.check_conf(correct_multiscale_config)

    assert result["multiscale"]["model"]["type"] == "pol"
    assert result["multiscale"]["model"]["degree"] == 2


@pytest.mark.parametrize(
    ["incorrect_multiscale_config"],
    [
        pytest.param(
            {
                "left_path": "tmp_repository_with_three_files",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2],
            },
            id="Repository without tif file",
        ),
    ],
    indirect=["incorrect_multiscale_config"],
)
def test_fails_with_different_number_of_tif(incorrect_multiscale_config):
    """
    Test that check conf fails when pyramid repositories contain different number of tif files
    """

    with pytest.raises(ValueError) as exc_info:
        multiscale.check_configuration.check_conf(incorrect_multiscale_config)
    assert str(exc_info.value) == "Left and right pyramid repositories must contain the same number of tif files."


@pytest.mark.parametrize(
    ["incorrect_multiscale_config"],
    [
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1],
            },
            id="Less scale factors than tif files",
        ),
        pytest.param(
            {
                "left_path": "tmp_correct_repository",
                "right_path": "tmp_correct_repository",
                "model_type": "pol",
                "model_degree": 2,
                "output": "output_test",
                "json_file": "tmp_json_file",
                "scale_factors": [1, 2, 4],
            },
            id="More scale factors than tif files",
        ),
    ],
    indirect=["incorrect_multiscale_config"],
)
def test_fails_with_wrong_number_of_scale_factors(incorrect_multiscale_config):
    """
    Test that check_conf fails when pyramid repositories contain a number of images
    that differs from the number of scale factors
    """

    with pytest.raises(ValueError) as exc_info:
        multiscale.check_configuration.check_conf(incorrect_multiscale_config)
    assert (
        str(exc_info.value) == "There should be as many images in the pyramid repositories as there are scale factors."
    )
