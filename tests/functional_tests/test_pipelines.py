# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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

"""
Run pandora2d configurations from end to end.
"""

# pylint: disable=redefined-outer-name

import json
from copy import deepcopy
from typing import Dict

import pytest


def remove_extra_keys(extended: dict, reference: dict):
    """
    Removes the extra keys in the `extended` dictionary that are not present in the `reference` dictionary.

    :param extended: The dictionary that may contain extra keys.
    :type extended: dict
    :param reference: The reference dictionary that contains the desired keys.
    :type reference: dict
    :return: A copy of the `extended` dictionary with only the keys present in the `reference` dictionary.
    :rtype: dict

    :Example:

    >>> extended = {"a": 1, "b": 2, "c": 3}
    >>> reference = {"a": 1, "b": 2}
    >>> remove_extra_keys(extended, reference)
    {'a': 1, 'b': 2}
    """
    extended_copy = deepcopy(extended)
    keys_only_in_extended = extended_copy.keys() - reference.keys()
    for key in keys_only_in_extended:
        extended_copy.pop(key)
    for extended_key, extended_value in extended_copy.items():
        reference_value = reference[extended_key]
        if isinstance(extended_value, dict) and isinstance(reference_value, dict):
            extended_copy[extended_key] = remove_extra_keys(extended_value, reference_value)
    return extended_copy


class TestRemoveExtrakeys:
    """Various tests on remove_extra_keys function."""

    def test_is_subset_dict(self):
        """Dicts have common keys."""
        reference = {"a": 2, "c": 3}
        value = {"a": 2, "b": {"aa": 4}, "c": 3}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_is_subset_dict_but_not_same_order(self):
        """Order of keys should not have influence."""
        reference = {"a": 2, "b": {"aa": 4}}
        value = {"b": {"aa": 4}, "a": 2}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_nested_subsets(self):
        """Values are dict that and result should recursively equal to reference."""
        reference = {"b": {"aa": 4}, "c": {"cc": 3}}
        value = {"a": 2, "b": {"aa": 4}, "c": {"cc": 3}}

        result = remove_extra_keys(value, reference)

        assert result == reference

    def test_no_common_key(self):
        """No common key means all keys are extra keys to be removed."""
        reference = {"c": 2}
        value = {"a": 2, "b": {"aa": 4}}

        result = remove_extra_keys(value, reference)

        assert result == {}


def test_monoband_with_nodata_not_nan(run_pipeline, correct_input_cfg, correct_pipeline_without_refinement):
    """Test a configuration with monoband images."""
    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement}
    configuration["input"]["left"]["nodata"] = -9999

    run_dir = run_pipeline(configuration)

    with open(run_dir / "output" / "cfg" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"

    # Test for report
    with open(run_dir / "output" / "report.json", encoding="utf8") as report_file:
        report = json.load(report_file)

    assert report["statistics"]["disparity"].keys() == {"row", "col"}


@pytest.mark.xfail(reason="saved nan in nodata is not valid json and is not comparable to nan")
def test_monoband_with_nan_nodata(run_pipeline, correct_input_cfg, correct_pipeline_without_refinement):
    """Test a configuration with monoband images and left nodata set to NaN."""
    configuration = {**correct_input_cfg, **correct_pipeline_without_refinement}

    run_dir = run_pipeline(configuration)

    with open(run_dir / "output" / "cfg" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"


@pytest.mark.xfail(reason="Multiband is not managed")
def test_multiband(run_pipeline, correct_multiband_input_cfg, correct_pipeline_without_refinement):
    """Test a configuration with multiband images."""
    configuration: Dict[str, Dict] = {**correct_multiband_input_cfg, **correct_pipeline_without_refinement}

    run_dir = run_pipeline(configuration)

    with open(run_dir / "output" / "cfg" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    result = remove_extra_keys(output_config, configuration)

    assert result == configuration
    assert list(result["pipeline"].keys()) == list(configuration["pipeline"].keys()), "Pipeline order not respected"


def test_optical_flow_configuration(run_pipeline, correct_input_cfg, correct_pipeline_with_optical_flow):
    """Test optical_flow configuration has a window_size and a step identical to matching_cost step."""
    configuration: Dict[str, Dict] = {**correct_input_cfg, **correct_pipeline_with_optical_flow}
    configuration["pipeline"]["refinement"]["iterations"] = 1

    run_dir = run_pipeline(configuration)

    with open(run_dir / "output" / "cfg" / "config.json", encoding="utf8") as output_file:
        output_config = json.load(output_file)

    matching_cost_cfg = output_config["pipeline"]["matching_cost"]
    refinement_cfg = output_config["pipeline"]["refinement"]

    # Check window_size and step parameters
    assert matching_cost_cfg["window_size"] == refinement_cfg["window_size"]
    assert matching_cost_cfg["step"] == refinement_cfg["step"]
