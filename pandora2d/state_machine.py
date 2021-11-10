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


from typing import Dict
import logging

from transitions import Machine, MachineError
import xarray as xr


class Pandora2DMachine(Machine):
    """
    Pandora2DMacine class to create and use a state machine
    """

    _transitions_run = [
        {"trigger": "matching_cost", "source": "begin", "dest": "cost_volumes", "after": "matching_cost_run"},
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disp_maps", "after": "disp_maps_run"},
        {"trigger": "refinement", "source": "disp_maps", "dest": "disp_maps", "after": "refinement_run"},
    ]

    _transitions_check = [
        {"trigger": "matching_cost", "source": "begin", "dest": "cost_volumes", "after": "matching_cost_check_conf"},
        {"trigger": "disparity", "source": "cost_volumes", "dest": "disp_maps", "after": "disparity_check_conf"},
        {"trigger": "refinement", "source": "disp_maps", "dest": "disp_maps", "after": "refinement_check_conf"},
    ]

    def __init__(
        self,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        no_data: float = None,
        disp_min_x: int = None,
        disp_max_x: int = None,
        disp_min_y: int = None,
        disp_max_y: int = None,
    ) -> None:
        """
        Initialize Pandora2D Machine

        Args:
            img_left (xr.Dataset, optional): left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray. Defaults to None.
            img_right (xr.Dataset, optional): right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray. Defaults to None.
            no_data (float, optional): [description]. Defaults to None.
            disp_min_x (int, optional): minimal disparity for columns. Defaults to None.
            disp_max_x (int, optional): maximal disparity for columns. Defaults to None.
            disp_min_y (int, optional): minimal disparity for lines. Defaults to None.
            disp_max_y (int, optional): maximal disparity for lines. Defaults to None.
        """

        # Left image
        self.left_img: xr.Dataset = img_left
        # Right image
        self.right_img: xr.Dataset = img_right
        # no data
        self.no_data: float = no_data
        # Minimum disparity
        self.disp_min_x: int = disp_min_x
        self.disp_min_y: int = disp_min_y
        # Maximum disparity
        self.disp_max_x: int = disp_max_x
        self.disp_max_y: int = disp_max_y

        self.pipeline_cfg: Dict = {"pipeline": {}}

        # Define avalaible states
        states_ = ["begin", "cost_volumes", "disp_maps"]

        # Initialize a machine without any transition
        Machine.__init__(
            self,
            states=states_,
            initial="begin",
            transitions=None,
            auto_transitions=False,
        )

        logging.getLogger("transitions").setLevel(logging.WARNING)

    ###### checking configuration################

    def run_prepare(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        disp_min_x: int,
        disp_max_x: int,
        disp_min_y: int,
        disp_max_y: int,
    ) -> None:
        """
        Prepare the machine before running

        Args:
            img_left (xr.Dataset): left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
            img_right (xr.Dataset): right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
            disp_min_x (int, optional): minimal disparity for columns
            disp_max_x (int, optional): maximal disparity for columns
            disp_min_y (int, optional): minimal disparity for lines
            disp_max_y (int, optional): maximal disparity for lines
            cfg_pipeline (Dict[str, dict]): configuration
        """

        self.left_img = img_left
        self.right_img = img_right
        self.disp_min_x = disp_min_x
        self.disp_max_x = disp_max_x
        self.disp_min_y = disp_min_y
        self.disp_max_y = disp_max_y

        self.add_transitions(self._transitions_run)

    def run(self, input_step: str, cfg: Dict[str, dict]) -> None:
        """
        Run pandora 2D step by triggering the corresponding machine transition

        Args:
            input_step (str): step to trigger
            cfg (Dict[str, dict]):  pipeline configuration
        """
        try:
            if len(input_step.split(".")) != 1:
                self.trigger(input_step.split(".")[0], cfg, input_step)
            else:
                self.trigger(input_step, cfg, input_step)
        except (MachineError, KeyError):
            print(
                "\n A problem occurs during Pandora2D running " + input_step + ". Be sure of your sequencement step  \n"
            )
            raise

    def run_exit(self) -> None:
        """
        Clear transitions and return to state begin

        :return: None
        """
        self.remove_transitions(self._transitions_run)
        self.set_state("begin")

    def check_conf(self, cfg: Dict[str, dict]) -> None:
        """
        Check configuration and transitions

        Args:
            cfg (Dict[str, dict]): pipeline configuration
        """

        # Add transitions to the empty machine.
        self.add_transitions(self._transitions_check)

        for input_step in list(cfg):
            try:
                self.trigger(input_step, cfg, input_step)

            except (MachineError, KeyError):
                print(
                    "\n Problem during Pandora2D checking configuration steps sequencing. "
                    "Check your configuration file. \n"
                )
                raise

        # Remove transitions
        self.remove_transitions(self._transitions_check)

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
            if trans["trigger"] not in deleted_triggers:
                self.remove_transition(trans["trigger"])
                deleted_triggers.append(trans["trigger"])

    def matching_cost_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

    def disparity_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the disparity computation configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

    def refinement_check_conf(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Check the refinement configuration

        :param cfg: configuration
        :type cfg: dict
        :param input_step: current step
        :type input_step: string
        :return: None
        """

    ###### run pipeline configuration ################

    def matching_cost_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Matching cost computation

        Args:
            cfg (Dict[str, dict]): pipeline configuration
            input_step (str): step to trigger
        """

    def disp_maps_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """
        Disparity computation

        Args:
            cfg (Dict[str, dict]): pipeline configuration
            input_step (str): step to trigger
        """

    def refinement_run(self, cfg: Dict[str, dict], input_step: str) -> None:
        """

        Refinement computation

        Args:
            cfg (Dict[str, dict]): pipeline configuration
            input_step (str): step to trigger
        """
