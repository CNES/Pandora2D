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
Module for common base of all MatchingCost methods.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, cast, Mapping, Tuple
import copy

from numpy.typing import NDArray
import numpy as np
import xarray as xr
from json_checker import And, Checker

from pandora.margins import Margins
from pandora import matching_cost as pandora_matching_cost

import pandora2d.schema as cst_schema
from pandora2d.criteria import get_criteria_dataarray


class BaseMatchingCost(ABC):
    """MatchingCost base class."""

    def __init__(self, cfg: Dict) -> None:
        """
        Initialisation of matching_cost class

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: None
        """
        self._cfg = self.check_conf(cfg)

        self._method = self._cfg["matching_cost_method"]
        # Cast to int in order to help mypy because self.cfg is a Dict, and it can not know the type of step.
        self._step_row = cast(int, self._cfg["step"][0])
        self._step_col = cast(int, self._cfg["step"][1])
        self._window_size = cast(
            int, self._cfg["window_size"]
        )  # _window_size attribute required to compute HalfWindowMargins
        self._subpix = cast(int, self._cfg["subpix"])
        self._spline_order = cast(int, self._cfg["spline_order"])

        self.cost_volumes: Union[xr.Dataset, None] = None

    @property
    def schema(self):
        return {
            # Census is not expected to be used with Pandora2D
            "matching_cost_method": And(str, lambda x: x not in ["census"]),
            "window_size": And(int, lambda input: input > 0 and (input % 2) != 0),
            "step": cst_schema.STEP_SCHEMA,
            "spline_order": And(int, lambda y: 1 <= y <= 5),
            "subpix": And(int, lambda sp: sp in [1, 2, 4]),
        }

    @property
    def defaults(self):
        return {
            "window_size": 5,
            "subpix": 1,
            "step": [1, 1],
            "spline_order": 1,
        }

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """Check the matching cost configuration

        :param cfg: user_config for matching cost
        :type cfg: dict
        :return: cfg: global configuration
        :rtype: cfg: dict
        """
        updated_config = self._update_with_default_config_values(cfg)
        checker = Checker(self.schema)
        checker.validate(updated_config)

        return updated_config

    def _update_with_default_config_values(self, cfg: Dict):
        return {**self.defaults, **cfg}

    @property
    def cfg(self) -> Mapping[str, Union[str, int, List[int]]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        :rtype: cfg: dict
        """
        return self._cfg

    @property
    def step(self) -> List[int]:
        """
        Get step [row, col]

        :return: step: list with row & col step
        :rtype: step: list
        """
        return [self._step_row, self._step_col]

    @property
    def window_size(self) -> int:
        """
        Get window_size

        :return: window_size: window used to compute correlation
        :rtype: window_size: int
        """
        return self._window_size

    @staticmethod
    def allocate_cost_volumes(
        cost_volume_attr: dict,
        row: np.ndarray,
        col: np.ndarray,
        disp_range_row: np.ndarray,
        disp_range_col: np.ndarray,
        np_data: np.ndarray = None,
    ) -> xr.Dataset:
        """
        Allocate the cost volumes

        :param cost_volume_attr: the cost_volume's attributes
        :type cost_volume_attr: xr.Dataset
        :param row: dimension of the image (row)
        :type row: np.ndarray
        :param col: dimension of the image (columns)
        :type col: np.ndarray
        :param disp_range_row: rows disparity range.
        :type disp_range_row: np.ndarray
        :param disp_range_col: columns disparity range.
        :type disp_range_col: np.ndarray
        :param np_data: 4D numpy.ndarray og cost_volumes. Defaults to None.
        :type np_data: np.ndarray
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """

        # Create the cost volume
        if np_data is None:
            np_data = np.zeros((len(row), len(col), len(disp_range_row), len(disp_range_col)), dtype=np.float32)

        cost_volumes = xr.Dataset(
            {"cost_volumes": (["row", "col", "disp_row", "disp_col"], np_data)},
            coords={"row": row, "col": col, "disp_row": disp_range_row, "disp_col": disp_range_col},
        )

        cost_volumes.attrs = cost_volume_attr

        return cost_volumes

    def get_cv_row_col_coords(
        self, img_row_coordinates: NDArray, img_col_coordinates: NDArray, cfg: Dict
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute cost_volumes row and col coordinates according to image coordinates

        :param img_row_coordinates: row coordinates of left image
        :type img_row_coordinates: NDArray
        :param img_col_coordinates: col coordinates of left image
        :type img_col_coordinates: NDArray
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :return: a Tuple of np.ndarray that contains the right coordinates for row and col
        :rtype: Tuple[NDArray, NDArray]
        """

        # Get updated ROI left/up margin for get_coordinates() method
        # To get right coordinates in cost_volume when initial left_margin > cfg["ROI"]["col"]["first"]
        # or initial up_margin > cfg["ROI"]["row"]["first"]
        # We need to have left_margin = cfg["ROI"]["col"]["first"] and up_margin = cfg["ROI"]["row"]["first"]
        cfg_for_get_coordinates = BaseMatchingCost.cfg_for_get_coordinates(cfg)

        # Get correct coordinates to be sure to process the first point of ROI
        if "ROI" in cfg:
            col_coords = pandora_matching_cost.AbstractMatchingCost.get_coordinates(
                margin=cfg_for_get_coordinates["ROI"]["margins"][0],
                img_coordinates=img_col_coordinates,
                step=self._step_col,
            )

            row_coords = pandora_matching_cost.AbstractMatchingCost.get_coordinates(
                margin=cfg_for_get_coordinates["ROI"]["margins"][1],
                img_coordinates=img_row_coordinates,
                step=self._step_row,
            )
        else:
            row_coords = np.arange(img_row_coordinates[0], img_row_coordinates[-1] + 1, self._step_row)
            col_coords = np.arange(img_col_coordinates[0], img_col_coordinates[-1] + 1, self._step_col)

        return row_coords, col_coords

    def get_disp_row_coords(self, img_left: xr.Dataset, margins: Margins) -> NDArray:
        """
        Compute cost_volumes row disparity coordinates according to image disparities

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param margins: refinement margins
        :type margins: Margins
        :return: a Tuple of np.ndarray that contains the right coordinates for disparities
        :rtype: Tuple[NDArray, NDArray]
        """

        # Get min/max row disparity grids
        grid_min_row = img_left["row_disparity"].sel(band_disp="min").data
        grid_max_row = img_left["row_disparity"].sel(band_disp="max").data

        # Obtain absolute min and max row disparities
        min_row, max_row = pandora_matching_cost.AbstractMatchingCost.get_min_max_from_grid(grid_min_row, grid_max_row)

        # Add refinement margins to disparity grids if needed.
        if margins is not None:
            min_row -= margins.up
            max_row += margins.down

        # Array with all row disparities
        disps_row = pandora_matching_cost.AbstractMatchingCost.get_disparity_range(min_row, max_row, self._subpix)

        return disps_row

    def get_disp_col_coords(self, img_left: xr.Dataset, margins: Margins) -> NDArray:
        """
        Compute cost_volumes col disparity coordinates according to image disparities

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param margins: refinement margins
        :type margins: Margins
        :return: a Tuple of np.ndarray that contains the right coordinates for disparities
        :rtype: Tuple[NDArray, NDArray]
        """

        # Get min/max col disparity grids
        grid_min_col = img_left["col_disparity"].sel(band_disp="min").data
        grid_max_col = img_left["col_disparity"].sel(band_disp="max").data

        # Obtain absolute min and max col disparities
        min_col, max_col = pandora_matching_cost.AbstractMatchingCost.get_min_max_from_grid(grid_min_col, grid_max_col)

        # Add refinement margins to disparity grids if needed.
        if margins is not None:
            min_col -= margins.left
            max_col += margins.right

        # Array with all col disparities
        disps_col = pandora_matching_cost.AbstractMatchingCost.get_disparity_range(min_col, max_col, self._subpix)

        return disps_col

    @staticmethod
    def cfg_for_get_coordinates(cfg: Dict) -> Dict:
        """
        Return right configuration to give to get_coordinates or get_coordinates_2d methods.

        To get right coordinates in cost_volume when initial left_margin > cfg["ROI"]["col"]["first"]
        or initial up_margin > cfg["ROI"]["row"]["first"]
        We need to have left_margin = cfg["ROI"]["col"]["first"] and up_margin = cfg["ROI"]["row"]["first"]

        :param cfg: user configuration
        :type cfg: Dict
        :return: updated configuration to be sure to process the first point of ROI
                 when ROI margin > ROI first point (left or up)
        :rtype: Dict
        """

        new_cfg = copy.deepcopy(cfg)

        if "ROI" in cfg:
            new_cfg["ROI"]["margins"] = (
                min(cfg["ROI"]["margins"][0], cfg["ROI"]["col"]["first"]),
                min(cfg["ROI"]["margins"][1], cfg["ROI"]["row"]["first"]),
                cfg["ROI"]["margins"][2],
                cfg["ROI"]["margins"][3],
            )
        return new_cfg

    @property
    @abstractmethod
    def margins(self): ...

    def allocate(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cfg: Dict,
        margins: Margins = None,
    ) -> None:
        """

        Allocate the cost volume

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param cfg: matching_cost computation configuration
        :type cfg: Dict
        :param margins: refinement margins
        :type margins: Margins
        :return: None
        """

        img_row_coordinates = img_left["im"].coords["row"].values
        img_col_coordinates = img_left["im"].coords["col"].values

        row_coords, col_coords = self.get_cv_row_col_coords(img_row_coordinates, img_col_coordinates, cfg)
        # Get disparity coordinates for cost_volumes
        disps_row_coords = self.get_disp_row_coords(img_left, margins)
        disps_col_coords = self.get_disp_col_coords(img_left, margins)

        grid_attrs = img_left.attrs

        grid_attrs.update(
            {
                "window_size": self._window_size,
                "subpixel": self._subpix,
                "offset_row_col": int((self._window_size - 1) / 2),
                "measure": self._method,
                "type_measure": "max",
                "disparity_margins": margins,
                "step": self.step,
            }
        )

        # Allocate 4D cost_volumes
        self.cost_volumes = self.allocate_cost_volumes(
            grid_attrs, row_coords, col_coords, disps_row_coords, disps_col_coords, None
        )

        self.cost_volumes["criteria"] = get_criteria_dataarray(img_left, img_right, self.cost_volumes)

    @abstractmethod
    def compute_cost_volumes(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        margins: Margins = None,
    ) -> xr.Dataset:
        """

        Computes the cost volumes

        :param img_left: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xr.Dataset
        :param img_right: xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xr.Dataset
        :param margins: refinement margins
        :type margins: Margins
        :return: cost_volumes: 4D Dataset containing the cost_volumes
        :rtype: cost_volumes: xr.Dataset
        """
