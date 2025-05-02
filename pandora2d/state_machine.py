#!/usr/bin/env python
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains class associated to the pandora state machine
"""

import copy
import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, TypedDict, Union

import xarray as xr
from typing_extensions import Annotated

try:
    import graphviz  # pylint: disable=unused-import

    # In order de avoid this message from Mypy:
    # Incompatible import of "Machine" \
    # (imported name has type "type[Machine]", local name has type "type[GraphMachine]")
    if TYPE_CHECKING:  # Mypy sees this:
        from transitions import Machine
    else:  # But we actually do this:
        from transitions.extensions import GraphMachine as Machine
except ImportError:
    from transitions import Machine

from transitions import MachineError
from pandora.margins import GlobalMargins

from pandora2d import common, disparity, estimation, img_tools, refinement, criteria
from pandora2d.matching_cost import MatchingCostRegistry, BaseMatchingCost
from pandora2d.profiling import mem_time_profile


class MarginsProperties(TypedDict):
    """Properties of Margins used in Margins transitions."""

    type: Literal["aggregate", "maximum"]
    margins: Annotated[List[int], '["left, "up", "right", "down"]']


class Pandora2DMachine(Machine):  # pylint:disable=too-many-instance-attributes
    """
    Pandora2DMachine class to create and use a state machine
    """

    _transitions_run = [
        {"trigger": "estimation", "source": "begin", "dest": "assumption", "before": "estimation_run"},
        {
            "trigger": "matching_cost",
            "source": "begin",
            "dest": "cost_volumes",
            "prepare": "matching_cost_prepare",
            "before": "matching_cost_run",
        },
        {
            "trigger": "matching_cost",
            "source": "assumption",
            "dest": "cost_volumes",
            "prepare": "matching_cost_prepare",
            "before": "matching_cost_run",
        },
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disparity_map", "before": "disparity_run"},
        {"trigger": "refinement", "source": "disparity_map", "dest": "disparity_map", "before": "refinement_run"},
    ]

    _transitions_check = [
        {"trigger": "estimation", "source": "begin", "dest": "assumption", "before": "estimation_check_conf"},
        {"trigger": "matching_cost", "source": "begin", "dest": "cost_volumes", "before": "matching_cost_check_conf"},
        {
            "trigger": "matching_cost",
            "source": "assumption",
            "dest": "cost_volumes",
            "before": "matching_cost_check_conf",
        },
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disparity_map", "before": "disparity_check_conf"},
        {
            "trigger": "refinement",
            "source": "disparity_map",
            "dest": "disparity_map",
            "before": "refinement_check_conf",
        },
    ]

    def __init__(
        self,
    ) -> None:
        """
        Initialize Pandora2D Machine

        """

        # Left image
        self.left_img: Optional[xr.Dataset] = None
        # Right image
        self.right_img: Optional[xr.Dataset] = None

        self.pipeline_cfg: Dict = {"pipeline": {}}
        self.completed_cfg: Dict = {}
        self.cost_volumes: xr.Dataset = xr.Dataset()
        self.dataset_disp_maps: xr.Dataset = xr.Dataset()

        # For communication between matching_cost and refinement steps
        self.step: list = None
        self.window_size: int = None
        self.margins_img = GlobalMargins()
        self.margins_disp = GlobalMargins()

        # Define available states
        states_ = ["begin", "assumption", "cost_volumes", "disparity_map"]

        # Instance matching_cost
        self.matching_cost_: Union[BaseMatchingCost, None] = None

        # Initialize a machine without any transition
        Machine.__init__(
            self,
            states=states_,
            initial="begin",
            transitions=None,
            auto_transitions=False,
        )

        logging.getLogger("transitions").setLevel(logging.WARNING)

    def run_prepare(self, img_left: xr.Dataset, img_right: xr.Dataset, cfg: dict) -> None:
        """
        Prepare the machine before running

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cfg: configuration
        :type cfg: Dict[str, dict]
        """

        self.left_img = img_left
        self.right_img = img_right
        self.completed_cfg = copy.copy(cfg)

        self.add_transitions(self._transitions_run)

    def run(self, input_step: str, cfg: Dict[str, dict]) -> None:
        """
        Run pandora 2D step by triggering the corresponding machine transition

        :param input_step: step to trigger
        :type input_step: str
        :param cfg: pipeline configuration
        :type  cfg: dict
        :return: None
        """
        try:
            if len(input_step.split(".")) != 1:
                self.trigger(input_step.split(".")[0], cfg, input_step)
            else:
                self.trigger(input_step, cfg, input_step)
        except (MachineError, KeyError, AttributeError):
            logging.error("Problem occurs during Pandora2D running %s. Be sure of your sequencement step", input_step)
            raise

    def run_exit(self) -> None:
        """
        Clear transitions and return to state begin

        :return: None
        """
        self.remove_transitions(self._transitions_run)  # type: ignore
        self.set_state("begin")

    def check_conf(self, cfg: Dict[str, dict]) -> None:
        """
        Check configuration and transitions

        :param cfg: pipeline configuration
        :type  cfg: dict
        :return:
        """

        # Add transitions to the empty machine.
        self.add_transitions(self._transitions_check)

        for input_step in list(cfg["pipeline"]):
            try:
                self.trigger(input_step, cfg, input_step)
            except (MachineError, KeyError, AttributeError):
                logging.error(
                    "Problem occurs during Pandora2D running %s. Be sure of your sequencement step", input_step
                )
                raise

        # Remove transitions
        self.remove_transitions(self._transitions_check)  # type: ignore

        # Coming back to the initial state
        self.set_state("begin")

    def remove_transitions(self, transition_list: Dict[str, dict]) -> None:
        """
        Delete all transitions defined in the input list

        :param transition_list: list of transitions
        :type transition_list: dict
        :return: None
        """
        # Transition is removed using trigger name. But one trigger name can be used by multiple transitions
        # In this case, the "remove_transition" function removes all transitions using this trigger name
        # deleted_triggers list is used to avoid multiple call of "remove_transition" with the same trigger name.
        deleted_triggers = []
        for trans in transition_list:
            if trans["trigger"] not in deleted_triggers:  # type: ignore
                self.remove_transition(trans["trigger"])  # type: ignore
                deleted_triggers.append(trans["trigger"])  # type: ignore

    def estimation_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the estimation computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        estimation_ = estimation.AbstractEstimation(cfg["pipeline"][input_step])  # type: ignore[abstract]
        self.pipeline_cfg["pipeline"][input_step] = estimation_.cfg

    def matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        MatchingCost = MatchingCostRegistry.get(  # pylint:disable=invalid-name # NOSONAR
            cfg["pipeline"][input_step]["matching_cost_method"]
        )
        matching_cost = MatchingCost(cfg["pipeline"][input_step])
        self.pipeline_cfg["pipeline"][input_step] = matching_cost.cfg
        self.step = matching_cost.step
        self.window_size = matching_cost.window_size
        self.margins_img.add_cumulative(input_step, matching_cost.margins)

    def disparity_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        disparity_ = disparity.Disparity(cfg["pipeline"][input_step])
        self.pipeline_cfg["pipeline"][input_step] = disparity_.cfg
        self.margins_img.add_cumulative(input_step, disparity_.margins)

    def refinement_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the refinement configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        refinement_ = refinement.AbstractRefinement(
            cfg["pipeline"][input_step], self.step, self.window_size
        )  # type: ignore[abstract]
        self.pipeline_cfg["pipeline"][input_step] = refinement_.cfg
        self.margins_disp.add_non_cumulative(input_step, refinement_.margins)

    def matching_cost_prepare(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Matching cost prepare

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        MatchingCost = MatchingCostRegistry.get(  # pylint:disable=invalid-name # NOSONAR
            cfg["pipeline"][input_step]["matching_cost_method"]
        )
        self.matching_cost_ = MatchingCost(cfg["pipeline"][input_step])

        self.matching_cost_.allocate(self.left_img, self.right_img, cfg, self.margins_disp.get("refinement"))

    @mem_time_profile(name="Estimation step")
    def estimation_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Shift's estimation step

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """

        logging.info("Estimation computation...")
        estimation_ = estimation.AbstractEstimation(cfg["pipeline"][input_step])  # type: ignore[abstract]

        row_disparity, col_disparity, shifts, extra_dict = estimation_.compute_estimation(self.left_img, self.right_img)

        self.completed_cfg = estimation_.update_cfg_with_estimation(
            cfg, col_disparity, row_disparity, shifts, extra_dict
        )

        # Update ROI margins with correct disparities
        roi = None
        if "ROI" in cfg:
            roi = img_tools.get_roi_processing(cfg["ROI"], cfg["input"]["col_disparity"], cfg["input"]["row_disparity"])
            # Recreate left and right image datasets with correct disparities and ROI margins
            self.left_img, self.right_img = img_tools.create_datasets_from_inputs(
                input_config=cfg["input"], roi=roi, estimation_cfg=None
            )
        else:
            # Update disparities for left and right image datasets
            self.left_img = img_tools.add_disparity_grid(self.left_img, col_disparity, row_disparity)
            self.right_img = img_tools.add_disparity_grid(self.right_img, col_disparity, row_disparity)

    @mem_time_profile(name="Matching cost step")
    def matching_cost_run(self, _, __) -> None:
        """
        Matching cost computation

        :return: None
        """

        logging.info("Matching cost computation...")

        self.cost_volumes = self.matching_cost_.compute_cost_volumes(
            self.left_img,
            self.right_img,
            self.margins_disp.get("refinement"),
        )

    @mem_time_profile(name="Disparity step")
    def disparity_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Disparity computation and validity mask

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """

        logging.info("Disparity computation...")
        disparity_ = disparity.Disparity(cfg["pipeline"][input_step])

        map_col, map_row, correlation_score = disparity_.compute_disp_maps(self.cost_volumes)

        dataset_validity = criteria.get_validity_dataset(self.cost_volumes["criteria"])

        self.dataset_disp_maps = common.dataset_disp_maps(
            map_row,
            map_col,
            self.cost_volumes.coords,
            correlation_score,
            dataset_validity,
            {
                "offset": {
                    "row": cfg.get("ROI", {}).get("row", {}).get("first", 0),
                    "col": cfg.get("ROI", {}).get("col", {}).get("first", 0),
                },
                "step": {
                    "row": cfg["pipeline"]["matching_cost"]["step"][0],
                    "col": cfg["pipeline"]["matching_cost"]["step"][1],
                },
                "invalid_disp": cfg["pipeline"]["disparity"]["invalid_disparity"],
                "crs": self.left_img.crs,
                "transform": self.left_img.transform,
            },
        )

        cv_coords = (self.cost_volumes.row.values, self.cost_volumes.col.values)

        criteria.apply_peak_on_edge(
            self.dataset_disp_maps["validity"],
            self.left_img,
            cv_coords,
            self.dataset_disp_maps["row_map"].data,
            self.dataset_disp_maps["col_map"].data,
        )

    @mem_time_profile(name="Refinement step")
    def refinement_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Subpixel disparity refinement

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """

        logging.info("Refinement computation...")

        if cfg["pipeline"][input_step]["refinement_method"] == "optical_flow":
            logging.warning("The optical flow method is still in an experimental phase.")
            logging.warning("The correlation score map is at a disparity level for the optical flow method.")

        refinement_run = refinement.AbstractRefinement(
            cfg["pipeline"][input_step], self.step, self.window_size
        )  # type: ignore[abstract]

        refine_map_col, refine_map_row, correlation_score = refinement_run.refinement_method(
            self.cost_volumes, self.dataset_disp_maps, self.left_img, self.right_img
        )
        self.dataset_disp_maps["row_map"].data = refine_map_row
        self.dataset_disp_maps["col_map"].data = refine_map_col
        self.dataset_disp_maps["correlation_score"].data = correlation_score
