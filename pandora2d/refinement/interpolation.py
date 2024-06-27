#!/usr/bin/env python
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2024 CS GROUP France
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
from typing import Tuple
from json_checker import And

from pandora.margins.descriptors import UniformMargins
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

    # pylint: disable=line-too-long
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html#scipy.interpolate.interp2d # NOSONAR
    # The minimum number of data points required along the interpolation axis is (k+1)**2,
    # with k=1 for linear, k=3 for cubic and k=5 for quintic interpolation.
    margins = UniformMargins(3)  # cubic kernel
    schema = {
        "refinement_method": And(str, lambda x: x in ["interpolation"]),
    }

    @staticmethod
    def wrapper_interp2d(params: np.ndarray, func: interp2d) -> np.ndarray:
        """

        Unpack tuple of arguments from minimize to fit in interp2d
        :param params: points coordinates
        :type params: np.ndarray
        :param func: interp2d scipy function
        :type func: scipy.interpolate.interpolate.interp2d
        :return: minimum of interp2d functions at points
        :rtype: np.ndarray
        """
        x, y = params
        return func(x, y)

    def compute_cost_matrix(self, p_args) -> Tuple[float, float, float]:
        """
        Process the interpolation and minimize of a cost_matrix
        :param cost_volumes: Dataset with 4D datas
        :type cost_volumes: xr.Dataset
        :param coords_pix_row: array from disp_min_row to disp_max_row
        :type coords_pix_row: np.ndarray
        :param coords_pix_col: array from disp_min_col to disp_max_col
        :type coords_pix_col: np.ndarray
        :param args_matrix_cost: 2D matrix with cost for one pixel (dim: dispy, dispx)
        :type args_matrix_cost: np.ndarray
        :return: res: min of args_matrix_cost in 2D
        :rtype: Tuple(float, float, float)
        """

        cost_volumes, coords_pix_row, coords_pix_col, args_matrix_cost = p_args

        # bounds ((disp_min_row, disp_max_row), (disp_min_col, disp_max_col))
        bounds = [
            (cost_volumes["disp_col"].data[0], cost_volumes["disp_col"].data[-1]),
            (cost_volumes["disp_row"].data[0], cost_volumes["disp_row"].data[-1]),
        ]
        # start point for minimize
        x_0 = (coords_pix_row, coords_pix_col)

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
            delta_col, delta_row, score_map = np.nan, np.nan, np.nan
        # if cost matrix with nans and cost
        else:
            if True in nans and np.all(nans) is not True:
                # interp nans values
                matrix_cost[nans] = np.interp(np.nonzero(nans)[0], np.nonzero(~nans)[0], matrix_cost[~nans])
            # interp matrix_cost
            interpolation2d_function = interp2d(
                cost_volumes["disp_col"].data, cost_volumes["disp_row"].data, matrix_cost, "cubic"
            )
            # looking for min
            delta_col, delta_row = minimize(
                self.wrapper_interp2d, args=(interpolation2d_function,), x0=x_0, bounds=bounds
            ).x
            score_map = abs(interpolation2d_function(delta_col, delta_row))[0]

        return delta_col, delta_row, score_map

    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute refine disparity maps
        :param cost_volumes: Cost_volumes has (row, col, disp_col, disp_row) dimensions
        :type cost_volumes: xr.Dataset
        :param disp_map: dataset of pixel disparity maps
        :type disp_map: xr.Dataset
        :param img_left: left image dataset
        :type img_left: xarray.Dataset
        :param img_right: right image dataset
        :type img_right: xarray.Dataset
        :return: delta_col, delta_row: subpixel disparity maps
        correlation score : matching_cost score
        :rtype: Tuple[np.array, np.array, np.array]
        """
        # cost_volumes data
        data = cost_volumes["cost_volumes"].data

        # transform 4D row, col, dcol, drow into drow, dcol, row * col
        nrow, ncol, ndispcol, ndisprow = data.shape
        cost_matrix = np.rollaxis(np.rollaxis(data, 3, 0), 3, 1).reshape((ndisprow, ndispcol, nrow * ncol))

        # flatten pixel maps for multiprocessing
        list_row = list(disp_map["row_map"].data.flatten().tolist())
        list_col = list(disp_map["col_map"].data.flatten().tolist())

        # args for multiprocessing
        args = [(cost_volumes, list_col[i], list_row[i], cost_matrix[:, :, i]) for i in range(0, cost_matrix.shape[2])]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            # liste([drow, dcol, score_col, score_row])
            map_carte = p.map(self.compute_cost_matrix, args)

        # compute disparity maps
        delta_col = np.array(map_carte)[:, 0]
        delta_row = np.array(map_carte)[:, 1]
        correlation_score = np.array(map_carte)[:, 2]

        # reshape disparity maps
        delta_col = np.reshape(delta_col, (disp_map["col_map"].data.shape[0], disp_map["col_map"].data.shape[1]))
        delta_row = np.reshape(delta_row, (disp_map["col_map"].data.shape[0], disp_map["col_map"].data.shape[1]))
        correlation_score = np.reshape(
            correlation_score, (disp_map["col_map"].data.shape[0], disp_map["col_map"].data.shape[1])
        )

        return delta_col, delta_row, correlation_score
