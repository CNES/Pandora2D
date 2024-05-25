#!/usr/bin/env python
#
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
This module contains functions associated to the optical flow method used in the refinement step.
"""
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from json_checker import And
from scipy.ndimage import map_coordinates
from pandora.margins import Margins

import pandora2d.schema as cst_schema
from . import refinement


@refinement.AbstractRefinement.register_subclass("optical_flow")
class OpticalFlow(refinement.AbstractRefinement):
    """
    OpticalFLow class allows to perform the subpixel cost refinement step
    """

    _iterations = None
    _invalid_disp = None

    _ITERATIONS = 4

    schema = {
        "refinement_method": And(str, lambda x: x in ["optical_flow"]),
        "iterations": And(int, lambda it: it > 0),
        "window_size": And(int, lambda input: input > 1 and (input % 2) != 0),
        "step": cst_schema.STEP_SCHEMA,
    }

    def __init__(self, cfg: dict = None, step: list = None, window_size: int = 5) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :param step: list containing row and col step
        :type step: list
        :param window_size: window size
        :type window_size: int
        :return: None
        """
        # Update user configuration with step and window_size parameters to check them
        cfg["window_size"] = window_size
        cfg["step"] = [1, 1] if step is None else step
        super().__init__(cfg)

        self._iterations = self.cfg["iterations"]
        self._refinement_method = self.cfg["refinement_method"]
        self._window_size = self.cfg["window_size"]
        self._step = self.cfg["step"]

    @classmethod
    def check_conf(cls, cfg: Dict) -> Dict:
        """
        Check the refinement configuration

        :param cfg: user_config for refinement
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """

        cfg["iterations"] = cfg.get("iterations", cls._ITERATIONS)

        cfg = super().check_conf(cfg)

        return cfg

    @property
    def margins(self):
        values = (self._window_size // 2 * ele for _ in range(2) for ele in self._step)
        return Margins(*values)

    def reshape_to_matching_cost_window(
        self,
        img: xr.Dataset,
        cost_volumes: xr.Dataset,
        disp_row: np.ndarray = None,
        disp_col: np.ndarray = None,
    ):
        """
        Transform image from (nb_col, nb_row) to (window_size, window_size, nbcol*nbrow)

        :param img: image to reshape
        :type img: xr.Dataset
        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_row: array dim [] containing all the row shift
        :type disp_row: np.ndarray
        :param disp_col: array dim [] containing all the columns shift
        :type disp_col: np.ndarray
        :return: array containing reshaped image [window_size, window_size, nbcol*nbrow]
        :rtype: np.ndarray
        """

        # get numpy array datas for image
        img_data = img["im"].data

        offset = max(self.margins.astuple())

        computable_col = cost_volumes.col.data[offset:-offset]
        computable_row = cost_volumes.row.data[offset:-offset]

        one_dim_size = len(computable_row) * len(computable_col)

        if disp_row is None and disp_col is None:
            patches = np.lib.stride_tricks.sliding_window_view(img_data, [self._window_size, self._window_size])
            patches = patches.reshape((one_dim_size, self._window_size, self._window_size)).transpose((1, 2, 0))
        else:
            # initiate values for right reshape computation
            offset = max(self.margins.astuple())
            patches = np.ndarray((self._window_size, self._window_size, one_dim_size))
            idx = 0

            for row in computable_row:
                for col in computable_col:
                    shift_col = (
                        0 if np.isnan(disp_col[idx]) or disp_col[idx] == self._invalid_disp else int(disp_col[idx])
                    )
                    shift_row = (
                        0 if np.isnan(disp_row[idx]) or disp_row[idx] == self._invalid_disp else int(disp_row[idx])
                    )

                    # get right pixel with his matching cost window
                    patch = img_data[
                        row - offset + shift_row : row + offset + 1 + shift_row,
                        col - offset + shift_col : col + offset + 1 + shift_col,
                    ]

                    # stock matching_cost window
                    if patch.shape == (self._window_size, self._window_size):
                        patches[:, :, idx] = patch
                    else:
                        patches[:, :, idx] = np.ones([self._window_size, self._window_size]) * np.nan

                    idx += 1

        return patches

    def warped_img(
        self, right_reshape: np.ndarray, delta_row: np.ndarray, delta_col: np.ndarray, index_to_compute: list
    ):
        """
        Shifted matching_cost window with computed disparity

        :param right_reshape: image right reshaped with dims (window_size, window_size, nbcol*nb_row)
        :type right_reshape: np.ndarray
        :param delta_row: rows disparity map
        :type delta_row: np.ndarray
        :param delta_col: columns disparity map
        :type delta_col: np.ndarray
        :param index_to_compute: list containing all valid pixel for computing optical flow
        :type index_to_compute: list
        :return: new array containing shifted matching_cost windows
        :rtype: np.ndarray
        """

        x, y = np.meshgrid(range(self._window_size), range(self._window_size))

        new_img = np.empty_like(right_reshape)

        # resample matching cost right windows
        for idx in index_to_compute:
            shifted_img = map_coordinates(
                right_reshape[:, :, idx], [y - delta_row[idx], x - delta_col[idx]], order=5, mode="reflect"
            )

            new_img[:, :, idx] = shifted_img

        return new_img

    def lucas_kanade_core_algorithm(self, left_data: np.ndarray, right_data: np.ndarray) -> Tuple[float, float]:
        """
        Implement lucas & kanade algorithm core

        :param left_data: matching_cost window for one pixel from left image
        :type left_data: np.ndarray
        :param right_data: matching_cost window for one pixel from left image
        :type right_data: np.ndarray
        :return: sub-pixel disparity computed by Lucas & Kanade optical flow
        :rtype: Tuple[float, float]
        """

        grad_y, grad_x = np.gradient(left_data)
        grad_t = right_data - left_data

        # Create A (grad_matrix) et B (time_matrix) matrix for Lucas Kanade
        grad_matrix = np.vstack((grad_x.flatten(), grad_y.flatten())).T
        time_matrix = grad_t.flatten()

        # Apply least-squares to solve the matrix equation AV= B where A is matrix containing partial derivate of (x,y)
        # B the matrix of partial derivate of t and V the motion we want to find

        try:
            motion = np.linalg.lstsq(grad_matrix, time_matrix, rcond=None)[0]
        # if matrix is full of NaN or 0
        except np.linalg.LinAlgError:
            motion = (self._invalid_disp, self._invalid_disp)

        return motion[1], motion[0]

    def optical_flow(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        list_idx_to_compute: list,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Computing optical flow between left and right image

        :param left_img: reshaped left image array
        :type left_img: np.ndarray
        :param right_img: reshaped right image array
        :type right_img: np.ndarray
        :param list_idx_to_compute: list of valid pixel
        :type list_idx_to_compute: list
        :return: computed sub-pixel disparity map
        :rtype: Tuple[np.ndarray, np.ndarray, list]
        """

        new_list_to_compute = []

        final_dec_row = np.zeros(left_img.shape[2])
        final_dec_col = np.zeros(left_img.shape[2])

        for idx in list_idx_to_compute:

            left_matching_cost = left_img[:, :, idx]
            right_matching_cost = right_img[:, :, idx]

            computed_delta_row, computed_delta_col = self.lucas_kanade_core_algorithm(
                left_matching_cost, right_matching_cost
            )

            # hypothesis from algorithm: shifts are < 1
            if abs(computed_delta_col) < 1 and abs(computed_delta_row) < 1:
                new_list_to_compute.append(idx)
            else:
                if abs(computed_delta_col) > 1:
                    computed_delta_col = 0
                if abs(computed_delta_row) > 1:
                    computed_delta_row = 0

            final_dec_row[idx] = computed_delta_row
            final_dec_col[idx] = computed_delta_col

        return final_dec_row, final_dec_col, new_list_to_compute

    def refinement_method(
        self, cost_volumes: xr.Dataset, disp_map: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the subpixel disparity maps

        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param disp_map: pixels disparity maps
        :type disp_map: xarray.Dataset
        :param img_left: left image dataset
        :type img_left: xarray.Dataset
        :param img_right: right image dataset
        :type img_right: xarray.Dataset
        :return: the refined disparity maps and disparity correlation score
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        # get invalid_disp value
        self._invalid_disp = disp_map.attrs["invalid_disp"]

        # get offset
        offset = max(self.margins.astuple())

        # get displacement map from disparity state
        initial_delta_row = disp_map["row_map"].data
        initial_delta_col = disp_map["col_map"].data

        delta_col = initial_delta_col[offset:-offset, offset:-offset].flatten()
        delta_row = initial_delta_row[offset:-offset, offset:-offset].flatten()

        # reshape left and right datas
        # from (nbcol, nbrow) to (window_size, window_size, nbcol*nbrow)
        reshaped_left = self.reshape_to_matching_cost_window(img_left, cost_volumes)
        reshaped_right = self.reshape_to_matching_cost_window(
            img_right,
            cost_volumes,
            delta_row,
            delta_col,
        )

        idx_to_compute = np.arange(reshaped_left.shape[2]).tolist()

        for _ in range(self._iterations):
            computed_drow, computed_dcol, idx_to_compute = self.optical_flow(
                reshaped_left, reshaped_right, idx_to_compute
            )

            reshaped_right = self.warped_img(reshaped_right, computed_drow, computed_dcol, idx_to_compute)

            # Pandora convention is left - d = right
            # Lucas&Kanade convention is left + d = right
            delta_col = delta_col - computed_dcol
            delta_row = delta_row - computed_drow

        # get finals disparity map dimensions
        nb_row, nb_col = initial_delta_col.shape
        nb_valid_points_row = nb_row - 2 * offset
        nb_valid_points_col = nb_col - 2 * offset

        delta_col = delta_col.reshape([nb_valid_points_row, nb_valid_points_col])
        delta_row = delta_row.reshape([nb_valid_points_row, nb_valid_points_col])

        # add borders
        delta_col = np.pad(delta_col, pad_width=offset, constant_values=self._invalid_disp)
        delta_row = np.pad(delta_row, pad_width=offset, constant_values=self._invalid_disp)

        delta_col[delta_col <= img_left.attrs["col_disparity_source"][0]] = self._invalid_disp
        delta_col[delta_col >= img_left.attrs["col_disparity_source"][1]] = self._invalid_disp
        delta_row[delta_row <= img_left.attrs["row_disparity_source"][0]] = self._invalid_disp
        delta_row[delta_row >= img_left.attrs["row_disparity_source"][1]] = self._invalid_disp

        return delta_col, delta_row, disp_map["correlation_score"].data
