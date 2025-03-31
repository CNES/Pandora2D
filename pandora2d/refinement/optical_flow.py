#!/usr/bin/env python
#
# Copyright (c) 2025 CS GROUP France
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

from typing import Dict, Tuple, List

import numpy as np
import xarray as xr
from json_checker import And
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from pandora.margins import Margins

import pandora2d.schema as cst_schema
from . import refinement


@refinement.AbstractRefinement.register_subclass("optical_flow")
class OpticalFlow(refinement.AbstractRefinement):
    """
    OpticalFLow class allows to perform the subpixel cost refinement step
    """

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
        coordinates: Tuple[List, List],
        disp_row: np.ndarray = None,
        disp_col: np.ndarray = None,
    ):
        """
        Transform image from (nb_col, nb_row) to (window_size, window_size, nbcol*nbrow)

        :param img: image to reshape
        :type img: xr.Dataset
        :param cost_volumes: cost_volumes 4D row, col, disp_col, disp_row
        :type cost_volumes: xarray.Dataset
        :param coordinates: min and max index coordinate for row and col [(first_row,last_row),(first_col,last_col)]
        :type coordinates: tuple
        :param disp_row: array dim [] containing all the row shift
        :type disp_row: np.ndarray
        :param disp_col: array dim [] containing all the columns shift
        :type disp_col: np.ndarray
        :return: array containing reshaped image [window_size, window_size, nbcol*nbrow]
        :rtype: np.ndarray
        """

        # get numpy array datas for image
        img_data = img["im"].data

        # get general offset value
        offset = cost_volumes.offset_row_col

        # get cost volume sub xarray with offset coordinates values
        offset_row, offset_col = coordinates
        cost_volumes_sub = cost_volumes.sel(
            row=slice(offset_row[0], offset_row[-1]), col=slice(offset_col[0], offset_col[-1])
        )

        # get computable cost volume data in row and col
        computable_col = cost_volumes_sub.col.data
        computable_row = cost_volumes_sub.row.data

        if disp_row is None and disp_col is None:
            # define image patches in one dim
            patches = np.lib.stride_tricks.sliding_window_view(img_data, [self._window_size, self._window_size])
            flattened_patches = patches.reshape(-1, self._window_size, self._window_size)

            # get patches id from original image
            id_patches_img = [
                int(c_row * img.sizes["col"]) + c_col
                for c_row in img["row"].data[offset:-offset]
                for c_col in img["col"].data[offset:-offset]
            ]

            # Associate each patches of the one dim image to the id of the true image patches
            patch_dict = {id_patches_img[i]: flattened_patches[i] for i in range(len(id_patches_img))}
            id_patches = [int(c_row * img.sizes["col"]) + c_col for c_row in computable_row for c_col in computable_col]

            # Filter patches to keep only id calculated with offset and step
            filtered_patches_list = [patch_dict[key] for key in id_patches if key in patch_dict]
            reshaped_patches = np.stack(filtered_patches_list, axis=-1).reshape(
                (self._window_size, self._window_size, len(filtered_patches_list))
            )
            return reshaped_patches

        # initiate values for right reshape computation
        offset = self._window_size // 2
        patches = np.ndarray((self._window_size, self._window_size, len(computable_row) * len(computable_col)))
        idx = 0

        for row in computable_row:
            for col in computable_col:
                shift_col = 0 if np.isnan(disp_col[idx]) or disp_col[idx] == self._invalid_disp else int(disp_col[idx])
                shift_row = 0 if np.isnan(disp_row[idx]) or disp_row[idx] == self._invalid_disp else int(disp_row[idx])

                # get right pixel with his matching cost window
                patch_row_start = row - offset + shift_row
                patch_row_end = row + offset + shift_row
                patch_col_start = col - offset + shift_col
                patch_col_end = col + offset + shift_col
                patch = img.sel(row=slice(patch_row_start, patch_row_end), col=slice(patch_col_start, patch_col_end))
                patch = patch["im"].data

                # stock matching_cost  window
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

    @staticmethod
    def find_nearest_column(value, data, direction):
        """
        Return the nearest column from initial column index coordinate in a given direction

        :param value: initial column index
        :type value: int
        :param data: cost volume coordinates
        :type data: np.ndarray
        :param direction: direction sign (must be + or -)
        :type direction: string
        """

        if direction == "+":
            return data[np.searchsorted(data, value, side="left")]
        if direction == "-":
            return data[np.searchsorted(data, value, side="right") - 1]

        raise ValueError("Direction must be '+' or '-'")

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
        offset = cost_volumes.offset_row_col

        # get first and last coordinates for row and col in cost volume dataset
        first_col_coordinate = cost_volumes.col.data[0] + offset
        last_col_coordinate = cost_volumes.col.data[-1] - offset
        col_extrema_coordinates = [
            self.find_nearest_column(first_col_coordinate, cost_volumes.col.data, "+"),
            self.find_nearest_column(last_col_coordinate, cost_volumes.col.data, "-"),
        ]

        first_row_coordinate = cost_volumes.row.data[0] + offset
        last_row_coordinate = cost_volumes.row.data[-1] - offset
        row_extrema_coordinates = [
            self.find_nearest_column(first_row_coordinate, cost_volumes.row.data, "+"),
            self.find_nearest_column(last_row_coordinate, cost_volumes.row.data, "-"),
        ]

        # get displacement map in row and col - from disparity min/max coordinates
        row_slice = slice(row_extrema_coordinates[0], row_extrema_coordinates[-1])
        col_slice = slice(col_extrema_coordinates[0], col_extrema_coordinates[-1])
        cost_volume_sub = cost_volumes.sel(row=row_slice, col=col_slice)
        disp_map_sub = disp_map.sel(row=cost_volume_sub.row, col=cost_volume_sub.col)
        delta_row = disp_map_sub["row_map"].data.flatten()
        delta_col = disp_map_sub["col_map"].data.flatten()

        # reshape left and right datas
        # from (nbcol, nbrow) to (window_size, window_size, nbcol*nbrow)
        reshaped_left = self.reshape_to_matching_cost_window(
            img_left, cost_volumes, (row_extrema_coordinates, col_extrema_coordinates)
        )
        reshaped_right = self.reshape_to_matching_cost_window(
            img_right,
            cost_volumes,
            (row_extrema_coordinates, col_extrema_coordinates),
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

        # get finals disparity map dimensions, add +1 because it began at 0
        nb_valid_points_row = int((row_extrema_coordinates[-1] - row_extrema_coordinates[0]) / cost_volumes.step[0] + 1)
        nb_valid_points_col = int((col_extrema_coordinates[-1] - col_extrema_coordinates[0]) / cost_volumes.step[1] + 1)

        delta_col = delta_col.reshape([nb_valid_points_row, nb_valid_points_col])
        delta_row = delta_row.reshape([nb_valid_points_row, nb_valid_points_col])

        # add border
        padding_top = (disp_map.sizes["row"] - delta_row.shape[0]) // 2
        padding_bottom = disp_map.sizes["row"] - delta_row.shape[0] - padding_top
        padding_left = (disp_map.sizes["col"] - delta_row.shape[1]) // 2
        padding_right = disp_map.sizes["col"] - delta_row.shape[1] - padding_left

        delta_row = np.pad(
            delta_row,
            pad_width=((padding_top, padding_bottom), (padding_left, padding_right)),
            constant_values=self._invalid_disp,
        )
        delta_col = np.pad(
            delta_col,
            pad_width=((padding_top, padding_bottom), (padding_left, padding_right)),
            constant_values=self._invalid_disp,
        )

        self._invalid_out_of_grid_disparities(cost_volumes.attrs["step"], delta_col, img_left["col_disparity"])
        self._invalid_out_of_grid_disparities(cost_volumes.attrs["step"], delta_row, img_left["row_disparity"])

        return delta_col, delta_row, disp_map["correlation_score"].data

    def _invalid_out_of_grid_disparities(self, step: List, delta: NDArray[np.floating], disparity: xr.DataArray):
        """
        Replace delta values by invalid_disp value when it is outside the corresponding disparity range defined by
        the disparity grid.

        :param step: [row_step, col_step]
        :type step: list
        :param delta: refined disparity map
        :type delta: np.NDArray
        :param disparity: pixelic disparity grids with min and max `band_disp` coordinates.
        :type disparity: xr.DataArray
        """
        delta[delta <= disparity.sel(band_disp="min").data[:: step[0], :: step[1]]] = self._invalid_disp
        delta[delta >= disparity.sel(band_disp="max").data[:: step[0], :: step[1]]] = self._invalid_disp
