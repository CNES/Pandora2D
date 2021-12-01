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
This module contains functions associated to the interpolation method used in the refinement step.
"""

import multiprocessing
from typing import Dict, Union, Tuple
from pandora.check_json import update_conf
from json_checker import And, Checker

from scipy.interpolate import interp2d
from scipy.optimize import minimize
import numpy as np
import xarray as xr

from . import refinement


@refinement.AbstractRefinement.register_subclass("interpolation")
class Interpolation(refinement.AbstractRefinement):
    """
    Interpolation class allows to perform the subpixel cost refinement step
    """

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        self.cfg = self.check_conf(**cfg)

    @staticmethod
    def check_conf(**cfg: Dict[str, Union[str, int]]) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Check the refinement configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        cfg = update_conf({"refinement_method": "interpolation"}, cfg)

        schema = {
            "refinement_method": And(str, lambda x: x in ["interpolation"]),
        }

        checker = Checker(schema)
        checker.validate(cfg)

        return cfg

    def compute_cost_matrix(self, p_args) -> Tuple[float, float]:
        """
        Process the interpolation and minimize of a cost_matrix
        :param cost_volumes: Dataset with 4D datas
        :type cost_volumes: xr.Dataset
        :param coords_pix_y: array from disp_min_y to disp_max_y
        :type coords_pix_y: np.array
        :param coords_pix_x: array from disp_min_x to disp_max_x
        :type coords_pix_x: np.array
        :param args_matrix_cost: 2D matrix with cost for one pixel (dim: dispy, dispx)
        :type args_matrix_cost: np.array
        :return: res: min of args_matrix_cost in 2D
        :rtype: Tuple(float, float)
        """

        cost_volumes, coords_pix_y, coords_pix_x, args_matrix_cost = p_args

        # bounds ((disp_min_y, disp_max_y), (disp_min_x, disp_max_x))
        bounds = [
            (cost_volumes["disp_col"].data[0], cost_volumes["disp_col"].data[-1]),
            (cost_volumes["disp_row"].data[0], cost_volumes["disp_row"].data[-1]),
        ]
        # start point for minimize
        x_0 = (coords_pix_y, coords_pix_x)

        # prepare cost_matrix for min or max research
        if cost_volumes.attrs["type_measure"] == "max":
            matrix_cost = -args_matrix_cost
        else:
            matrix_cost = args_matrix_cost

        # looking for inf values
        matrix_cost[matrix_cost == np.inf] = np.nan

        # looking for nans values
        nans = np.isnan(matrix_cost)

        # if matrix_cost full of nans
        if np.all(nans):
            res = (np.nan, np.nan)
        # if cost matrix with nans and cost
        elif True in nans and np.all(nans) is not True:
            # interp nans values
            matrix_cost[nans] = np.interp(np.nonzero(nans)[0], np.nonzero(~nans)[0], matrix_cost[~nans])
            # interp matrix_cost
            fonction_interpolation = interp2d(
                cost_volumes["disp_col"].data, cost_volumes["disp_row"].data, matrix_cost, "cubic"
            )
            wrap = lambda f: fonction_interpolation(*f)
            # looking for min
            res = minimize(wrap, x_0, bounds=bounds).x
        # if cost matrix full of values
        else:
            # interp matrix_cost
            fonction_interpolation = interp2d(
                cost_volumes["disp_col"].data, cost_volumes["disp_row"].data, matrix_cost, kind="cubic"
            )
            # looking for min
            wrap = lambda f: fonction_interpolation(*f)
            res = minimize(wrap, x_0, bounds=bounds).x

        return res

    def refinement_method(self, cost_volumes: xr.Dataset, pixel_maps: xr.Dataset) -> Tuple[np.array, np.array]:
        """
        Compute refine disparity maps
        :param cost_volumes: Cost_volumes has (row, col, disp_col, disp_row) dimensions
        :type cost_volumes: xr.Dataset
        :param pixel_maps: dataset of pixel disparity maps
        :type pixel_maps: xr.Dataset
        :return: delta_x, delta_y: subpixel disparity maps
        :rtype: Tuple[np.array, np.array]
        """
        #cost_columes data
        data = cost_volumes["cost_volumes"].data

        # transform 4D row, col, dcol, drow into drow, dcol, row * col
        nrow, ncol, ndispx, ndispy = data.shape
        cost_matrix = np.rollaxis(np.rollaxis(data, 3, 0), 3, 1).reshape((ndispy, ndispx, nrow * ncol))

        # flatten pixel maps for multiprocessing
        liste_y = list(pixel_maps["row_map"].data.flatten().tolist())
        liste_x = list(pixel_maps["col_map"].data.flatten().tolist())

        # args for multiprocessing
        args = [(cost_volumes, liste_x[i], liste_y[i], cost_matrix[:, :, i]) for i in range(0, cost_matrix.shape[2])]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            # liste([drow, dcol])
            mat_carte = p.map(self.compute_cost_matrix, args)

        # compute disparity maps
        delta_x = np.array(mat_carte)[:, 0]
        delta_y = np.array(mat_carte)[:, 1]

        # reshape disparity maps
        delta_x = np.reshape(delta_x, (pixel_maps["col_map"].data.shape[0], pixel_maps["col_map"].data.shape[1]))
        delta_y = np.reshape(delta_y, (pixel_maps["col_map"].data.shape[0], pixel_maps["col_map"].data.shape[1]))

        return delta_x, delta_y
