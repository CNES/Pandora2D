# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Module with global test fixtures.
"""

import pathlib
import pytest


def pytest_collection_modifyitems(config, items):
    """
    Update marker collections

    1. Adds the marker corresponding to the various directories ending in "_test".
    The various markers created are:
    - unit_tests
    - functional_tests
    - performance_tests
    - resource_tests
    2. Disables pytest_monitor.
    """
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        mark_name = next((part for part in rel_path.parts if part.endswith("_tests")), "")
        if mark_name:
            mark = getattr(pytest.mark, mark_name)
            item.add_marker(mark)
            item.add_marker(pytest.mark.monitor_skip_test)


@pytest.fixture()
def classic_config():
    return "./tests/data/json_conf_files/classic_cfg.json"
