# pylint:disable=line-too-long
#!/usr/bin/env python
#
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
This module contains functions to test the Pandora2D notebooks.
"""
import subprocess
import tempfile
import pytest


@pytest.mark.notebook_tests
class TestNotebooks:
    """
    Allows to test the pandora2d notebooks
    """

    def test_introduction_and_basic_usage(self):
        """
        Test that the introduction_and_basic_usage notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    f"jupyter nbconvert --to script notebooks/introduction_and_basic_usage.ipynb --output-dir {directory}"
                ],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/introduction_and_basic_usage.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    def test_usage_step_roi_config(self):
        """
        Test that the usage_step_roi_config notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [f"jupyter nbconvert --to script notebooks/usage_step_roi_config.ipynb --output-dir {directory}"],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/usage_step_roi_config.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    def test_usage_dichotomy(self):
        """
        Test that the usage_dichotomy notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [f"jupyter nbconvert --to script notebooks/usage_dichotomy.ipynb --output-dir {directory}"],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/usage_dichotomy.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    def test_estimation_step_explained(self):
        """
        Test that the estimation_step_explained notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [f"jupyter nbconvert --to script notebooks/estimation_step_explained.ipynb --output-dir {directory}"],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/estimation_step_explained.py"],
                shell=True,
                check=False,
                cwd="notebooks",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0

    def test_margins(self):
        """
        Test that the test_margins notebook runs without errors

        """
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    f"jupyter nbconvert --to script notebooks/advanced_examples/test_margins.ipynb --output-dir {directory}"
                ],
                shell=True,
                check=False,
            )
            out = subprocess.run(
                [f"ipython {directory}/test_margins.py"],
                shell=True,
                check=False,
                cwd="notebooks/advanced_examples",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            assert out.returncode == 0
