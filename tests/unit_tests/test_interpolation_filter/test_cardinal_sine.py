#  Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
"""
Test Cardinal sine filter.
"""

# Make pylint happy with fixtures:
# pylint: disable=redefined-outer-name, protected-access

import json_checker
import numpy as np
import pytest

from pandora.margins import Margins

import pandora2d.interpolation_filter


@pytest.fixture()
def size():
    return 6


@pytest.fixture()
def fractional_shift():
    return 0.25


@pytest.fixture()
def config(size):
    return {"method": "sinc", "size": size}


@pytest.fixture()
def filter_instance(config, fractional_shift):
    return pandora2d.interpolation_filter.AbstractFilter(  # pylint: disable=abstract-class-instantiated
        cfg=config,
        fractional_shift=fractional_shift,
    )  # type: ignore[abstract]


def test_factory(filter_instance):
    assert isinstance(filter_instance, pandora2d.interpolation_filter.cardinal_sine.CardinalSine)


class TestCheckConf:
    """Test the check_conf method."""

    def test_method_field(self, config):
        """An exception should be raised if `method` is not `sinc`."""
        config["method"] = "invalid_method"

        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            pandora2d.interpolation_filter.cardinal_sine.CardinalSine(config)
        assert "invalid_method" in err.value.args[0]

    @pytest.mark.parametrize("size", [5, 22])
    def test_out_of_bound_size_field(self, config):
        """An exception should be raised if `size` is not between 6 and 21."""
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            pandora2d.interpolation_filter.cardinal_sine.CardinalSine(config)
        assert "size" in err.value.args[0]

    def test_size_is_optional_and_default_value(self):
        """If size is not given into config it should default to 6."""
        config = {"method": "sinc"}
        sinc_filter = pandora2d.interpolation_filter.cardinal_sine.CardinalSine(config)
        assert sinc_filter._HALF_SIZE == 6
        assert sinc_filter._SIZE == 13


@pytest.mark.parametrize("fractional_shift", [-0.5, 1, 4])
def test_raise_error_with_invalid_fractional(config, fractional_shift):
    """Test an exception is raised if not in range [0,1[."""
    with pytest.raises(
        ValueError,
        match=f"fractional_shift greater than 0 and lower than 1 expected, got {fractional_shift}",
    ):
        pandora2d.interpolation_filter.AbstractFilter(  # pylint: disable=abstract-class-instantiated
            cfg=config,
            fractional_shift=fractional_shift,
        )  # type: ignore[abstract]


@pytest.mark.parametrize("size", [6, 21])
def test_margins(filter_instance, size):
    assert filter_instance.margins == Margins(size, size, size, size)


@pytest.fixture()
def reference_implementation(size, subpixel):
    """Reference implementation translated from Medicis."""
    sigma = size
    nb_of_coeffs_per_precision = 1 + (size * 2)
    tab_coeffs = np.zeros([subpixel, nb_of_coeffs_per_precision])

    aux1 = -2.0 / (sigma * sigma * np.pi)

    for i in range(subpixel):
        precision = i / subpixel

        for j in range(nb_of_coeffs_per_precision):
            aux = (precision - (j - size)) * np.pi

            if aux == 0:
                tab_coeffs[i][j] = 1
            else:
                tab_coeffs[i][j] = np.sin(aux) * np.exp(aux1 * aux * aux) / aux

        somme = np.sum(tab_coeffs[i])
        tab_coeffs[i] /= somme

    return tab_coeffs


@pytest.mark.parametrize("size", [6, 10, 21])
@pytest.mark.parametrize("subpixel", [4, 8, 16])
def test_compute_coefficient_table(reference_implementation, size, subpixel):
    """Test values computed against reference implementation."""
    fractional_shifts = np.arange(subpixel) / subpixel
    result = pandora2d.interpolation_filter.cardinal_sine.compute_coefficient_table(size, fractional_shifts)
    # Do to the use of `np.sinc` there is a little difference in the results so we use almost_equal
    np.testing.assert_array_almost_equal(result, reference_implementation)


def test_get_coeffs(filter_instance, size):
    """Test retrieve good coefficients from computed table."""
    fractional_shifts = np.arange(4) / 4
    coeffs = pandora2d.interpolation_filter.cardinal_sine.compute_coefficient_table(size, fractional_shifts)

    np.testing.assert_array_equal(filter_instance.get_coeffs(0.25), coeffs[1])
    np.testing.assert_array_equal(filter_instance.get_coeffs(0.5), coeffs[2])
    np.testing.assert_array_equal(filter_instance.get_coeffs(0.75), coeffs[3])
