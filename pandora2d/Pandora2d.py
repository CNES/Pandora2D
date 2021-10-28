#!/usr/bin/env python
# coding: utf8
#
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
This module contains the general function to run Pandora 2D pipeline.
"""

import argparse

import pandora2d


def get_parser():
    """
    ArgumentParser for Pandora 2D

    :return parser
    """

    parser = argparse.ArgumentParser(description="Pandora 2D")
    parser.add_argument(
        "config",
        help="path to a json file containing the input/output files paths and \
            algorithm parameters",
    )
    parser.add_argument("path_output", help="path to the output directory for disparity maps")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    return parser


def main():
    """
    Call Pandora2D's main
    """

    # Get parser
    parser = get_parser()
    args = parser.parse_args()

    # Run the Pandora 2D pipeline
    pandora2d.main(args.config, args.path_output, args.verbose)


if __name__ == "__main__":
    main()
