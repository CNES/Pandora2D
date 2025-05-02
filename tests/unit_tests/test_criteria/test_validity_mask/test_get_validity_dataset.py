#  Copyright (c) 2025. Centre National d'Etudes Spatiales (CNES).
#
#  This file is part of PANDORA2D
#
#      https://github.com/CNES/Pandora2D
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Check get_validity_dataset behavior.
"""

from random import choice

import numpy as np
import pytest
import xarray as xr

from pandora2d import criteria


@pytest.mark.parametrize(["criteria_var"], [pytest.param(c, id=c.name) for c in criteria.DISPARITY_DEPENDENT_CRITERIA])
class TestGetValidityDataset:
    """Check get_validity_dataset behavior."""

    @pytest.fixture()
    def criteria_dataarray(self):
        """An empty criteria_dataarray."""
        return xr.DataArray(
            data=np.zeros((3, 2, 3, 2), dtype=np.uint8),
            coords={
                "row": np.arange(0, 3),
                "col": np.arange(0, 2),
                "disp_row": np.arange(-1, 2),
                "disp_col": np.arange(0, 2),
            },
        )

    @pytest.fixture()
    def other_criteria_var(self, criteria_var):
        return choice(list(criteria.DISPARITY_DEPENDENT_CRITERIA - {criteria_var}))

    def test_no_criteria(self, criteria_dataarray, criteria_var):
        """validity_dataset should be full of zeros."""
        result = criteria.get_validity_dataset(criteria_dataarray)

        assert (result["validity"].sel(criteria="validity_mask") == 0).all()
        assert (result["validity"].sel(criteria=criteria_var.name) == 0).all()

    def test_empty_even_with_other_criteria(self, criteria_dataarray, criteria_var, other_criteria_var):
        """A criteria is not affected by presence of another one."""
        criteria_dataarray.loc[{"row": 2, "col": 1, "disp_row": -1, "disp_col": 1}] = np.uint8(other_criteria_var)
        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=2, col=1) == 1
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 1) == 1
        assert (result["validity"].sel(criteria=criteria_var.name) == 0).all()

    @pytest.mark.parametrize(
        "other_criteria_var", [pytest.param(c, id=c.name) for c in criteria.DISPARITY_INDEPENDENT_CRITERIA]
    )
    def test_empty_even_with_invalidating_criteria(self, criteria_dataarray, criteria_var, other_criteria_var):
        """invalidating criteria fills the cost_surface and thus invalidates point."""
        criteria_dataarray.loc[{"row": 2, "col": 1}] = np.uint8(other_criteria_var)
        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=2, col=1) == 2
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 2) == 1
        assert (result["validity"].sel(criteria=criteria_var.name) == 0).all()

    def test_only_one_disparity(self, criteria_dataarray, criteria_var):
        """Partial invalidity is raised when a Criteria is present for at least one disparity couple."""
        criteria_dataarray.loc[{"row": 1, "col": 0, "disp_row": 0, "disp_col": 0}] = np.uint8(criteria_var)

        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=1, col=0) == 1
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 1) == 1
        assert result["validity"].sel(criteria=criteria_var.name, row=1, col=0) == 1

    def test_multiple_disparities(self, criteria_dataarray, criteria_var):
        """Having a Criteria on multiple disparities does not change the result."""
        criteria_dataarray.loc[{"row": 1, "col": 0, "disp_row": [0, 1], "disp_col": 0}] = np.uint8(criteria_var)

        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=1, col=0) == 1
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 1) == 1
        assert result["validity"].sel(criteria=criteria_var.name, row=1, col=0) == 1

    def test_multiple_criteria(self, criteria_dataarray, criteria_var, other_criteria_var):
        """Having multiple Criteria on multiple disparities does not change the result."""
        criteria_dataarray.loc[{"row": 1, "col": 0, "disp_row": [0, 1], "disp_col": 0}] = np.uint8(
            criteria_var | other_criteria_var
        )

        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=1, col=0) == 1
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 1) == 1
        assert result["validity"].sel(criteria=criteria_var.name, row=1, col=0) == 1
        assert result["validity"].sel(criteria=other_criteria_var.name, row=1, col=0) == 1

    @pytest.mark.parametrize(
        "other_criteria_var", [pytest.param(c, id=c.name) for c in criteria.DISPARITY_INDEPENDENT_CRITERIA]
    )
    def test_combined_with_invalidating_criteria(self, criteria_dataarray, criteria_var, other_criteria_var):
        """invalidating criteria fills the cost_surface and thus invalidates point."""
        criteria_dataarray.loc[{"row": 1, "col": 0}] = np.uint8(other_criteria_var)
        criteria_dataarray.loc[{"row": 1, "col": 0, "disp_row": [0, 1], "disp_col": 0}] = np.uint8(
            criteria_var | other_criteria_var
        )

        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=1, col=0) == 2
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 2) == 1
        assert result["validity"].sel(criteria=criteria_var.name, row=1, col=0) == 1
        assert result["validity"].sel(criteria=other_criteria_var.name, row=1, col=0) == 1

    def test_invalidating(self, criteria_dataarray, criteria_var):
        """When all disparities of a point have a Criteria, the point is invalid."""
        criteria_dataarray.loc[{"row": 1, "col": 0}] = np.uint8(criteria_var)

        result = criteria.get_validity_dataset(criteria_dataarray)

        assert result["validity"].sel(criteria="validity_mask", row=1, col=0) == 2
        assert np.count_nonzero(result["validity"].sel(criteria="validity_mask") == 2) == 1
        assert result["validity"].sel(criteria=criteria_var.name, row=1, col=0) == 1
