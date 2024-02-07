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
This module contains class associated to the pandora state machine
"""

from typing import Dict, TYPE_CHECKING, List, TypedDict, Literal, Optional, Union
import logging
from operator import add
import numpy as np
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
from pandora2d import matching_cost, disparity, refinement, common


class MarginsProperties(TypedDict):
    """Properties of Margins used in Margins transitions."""

    type: Literal["aggregate", "maximum"]
    margins: Annotated[List[int], '["left, "up", "right", "down"]']


class Pandora2DMachine(Machine):
    """
    Pandora2DMacine class to create and use a state machine
    """

    _transitions_run = [
        {
            "trigger": "matching_cost",
            "source": "begin",
            "dest": "cost_volumes",
            "prepare": "matching_cost_prepare",
            "after": "matching_cost_run",
        },
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disp_maps", "after": "disp_maps_run"},
        {"trigger": "refinement", "source": "disp_maps", "dest": "disp_maps", "after": "refinement_run"},
    ]

    _transitions_check = [
        {"trigger": "matching_cost", "source": "begin", "dest": "cost_volumes", "after": "matching_cost_check_conf"},
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disp_maps", "after": "disparity_check_conf"},
        {"trigger": "refinement", "source": "disp_maps", "dest": "disp_maps", "after": "refinement_check_conf"},
    ]

    margins = GlobalMargins()

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
        # Column's min, max disparities
        self.disp_min_col: np.ndarray = None
        self.disp_max_col: np.ndarray = None
        # Row's min, max disparities
        self.disp_min_row: np.ndarray = None
        self.disp_max_row: np.ndarray = None

        self.pipeline_cfg: Dict = {"pipeline": {}}
        self.cost_volumes: xr.Dataset = xr.Dataset()
        self.dataset_disp_maps: xr.Dataset = xr.Dataset()

        # Define avalaible states
        states_ = ["begin", "cost_volumes", "disp_maps"]

        # Instance matching_cost
        self.matching_cost_: Union[matching_cost.MatchingCost, None] = None

        # Initialize a machine without any transition
        Machine.__init__(
            self,
            states=states_,
            initial="begin",
            transitions=None,
            auto_transitions=False,
        )

        logging.getLogger("transitions").setLevel(logging.WARNING)

    def run_prepare(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
    ) -> None:
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
        """

        self.left_img = img_left
        self.right_img = img_right
        # Column's min, max disparities
        self.disp_min_col = img_left["col_disparity"].sel(band_disp="min").data
        self.disp_max_col = img_left["col_disparity"].sel(band_disp="max").data
        # Row's min, max disparities
        self.disp_min_row = img_left["row_disparity"].sel(band_disp="min").data
        self.disp_max_row = img_left["row_disparity"].sel(band_disp="max").data

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

    def get_global_margins(self):
        """
        Aggregation margins
        """
        aggregate_margins = [0, 0, 0, 0]
        max_margins = [0, 0, 0, 0]
        for cfg in self._transitions_margins.values():
            step_margins = cfg["margins"]
            type_margin = cfg["type"]
            if type_margin == "aggregate":
                aggregate_margins = list(map(add, aggregate_margins, step_margins))
            else:
                max_margins = list(map(max, aggregate_margins, step_margins))

        return [max(x, y) for x, y in zip(max_margins, aggregate_margins)]

    def matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        matching_cost_ = matching_cost.MatchingCost(cfg["pipeline"][input_step])
        self.pipeline_cfg["pipeline"][input_step] = matching_cost_.cfg
        self.margins.add_cumulative(input_step, matching_cost_.margins)

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
        self.margins.add_cumulative(input_step, disparity_.margins)

    def refinement_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the refinement configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

        refinement_ = refinement.AbstractRefinement(cfg["pipeline"][input_step])  # type: ignore[abstract]
        self.pipeline_cfg["pipeline"][input_step] = refinement_.cfg
        self.margins.add_non_cumulative(input_step, refinement_.margins)

    def matching_cost_prepare(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Matching cost prepare

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """
        self.matching_cost_ = matching_cost.MatchingCost(cfg["pipeline"][input_step])
        self.matching_cost_.allocate_cost_volume_pandora(
            self.left_img, self.right_img, self.disp_min_col, self.disp_max_col, cfg["pipeline"][input_step]
        )

    def matching_cost_run(self, _, __) -> None:
        """
        Matching cost computation

        :return: None
        """

        logging.info("Matching cost computation...")

        self.cost_volumes = self.matching_cost_.compute_cost_volumes(
            self.left_img,
            self.right_img,
            self.disp_min_col,
            self.disp_max_col,
            self.disp_min_row,
            self.disp_max_row,
        )

    def disp_maps_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Disparity computation and validity mask

        :param cfg: pipeline configuration
        :type  cfg: dict
        :param input_step: step to trigger
        :type input_step: str
        :return: None
        """

        logging.info("Disparity computation...")
        disparity_run = disparity.Disparity(cfg["pipeline"][input_step])

        map_col, map_row = disparity_run.compute_disp_maps(self.cost_volumes)
        self.dataset_disp_maps = common.dataset_disp_maps(map_row, map_col)

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
        refinement_run = refinement.AbstractRefinement(cfg["pipeline"][input_step])  # type: ignore[abstract]

        refine_map_col, refine_map_row = refinement_run.refinement_method(self.cost_volumes, self.dataset_disp_maps)
        self.dataset_disp_maps = common.dataset_disp_maps(refine_map_row, refine_map_col)
