""" Module with global test fixtures. """

from copy import deepcopy

import pytest

from pandora2d import Pandora2DMachine


@pytest.fixture()
def pandora2d_machine():
    """pandora2d_machine"""
    return Pandora2DMachine()


@pytest.fixture()
def pipeline_config(correct_pipeline):
    return deepcopy(correct_pipeline)


@pytest.fixture(name="left_img_path")
def left_img_path_fixture():
    return "./tests/data/images/cones/left.png"


@pytest.fixture(name="right_img_path")
def right_img_path_fixture():
    return "./tests/data/images/cones/right.png"


@pytest.fixture
def correct_input_cfg(left_img_path, right_img_path):
    return {
        "input": {
            "left": {
                "img": left_img_path,
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
