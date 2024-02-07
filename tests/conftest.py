""" Module with global test fixtures. """

from copy import deepcopy

import pytest

from pandora2d import Pandora2DMachine
from tests import common


@pytest.fixture()
def pandora2d_machine():
    """pandora2d_machine"""
    return Pandora2DMachine()


@pytest.fixture()
def pipeline_config():
    return deepcopy(common.correct_pipeline)
