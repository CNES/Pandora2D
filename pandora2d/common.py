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
This module contains functions allowing to save the results and the configuration of Pandora pipeline.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Generic, List, Tuple, Type, TypeVar, Union
from os import PathLike

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pandora.common import write_data_array
from rasterio import Affine
from rasterio.crs import CRS

from pandora2d import reporting
from pandora2d.constants import Criteria
from pandora2d.img_tools import remove_roi_margins

# mypy: disable-error-code="attr-defined, no-redef"
# pylint: disable=useless-import-alias

# xarray.Coordinates corresponds to the latest version of xarray.
# xarray.Coordinate corresponds to the version installed by the artifactory.
# Try/except block to be deleted once the version of xarray has been updated by CNES.
try:
    from xarray import Coordinates as Coordinates
except ImportError:
    from xarray import Coordinate as Coordinates

T = TypeVar("T")


class Registry(Generic[T]):
    """Registry of classes.

    A class to decorate classes in order to register them with a string name.
    """

    def __init__(self, default: Union[Type[T], None] = None) -> None:
        """
        Initialize the registry with an optional default class.

        :param default: Default class to return if name is not registered. If None, will raise a KeyError.
        :type default: Union[Type[T], None]
        """
        self.registered: Dict[str, Type[T]] = {}
        self.default = default

    def add(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Register a class with `name`.

        :param name: Name to register the decorated class with.
        :type name: str
        :return: The decorated class.
        :rtype: Type[T]
        """

        def decorator(cls: Type[T]) -> Type[T]:
            """Returned decorator used for register class."""
            # No verification is made about already registered name: we need to decide on desired behavior
            self.registered[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """
        Get the class registered as `name`.

        :param name: The name of the registered class to retrieve.
        :type name: str
        :return: The class registered under `name` or the default class if not found.
        :rtype: Type[T]
        :raises KeyError: If no class is registered under `name` and no default is set.
        """
        if self.default is None and name not in self.registered:
            raise KeyError(f"No class registered with name `{name}`.")
        return self.registered.get(name, self.default)


class AllPrimitiveEncoder(json.JSONEncoder):
    """JSON Encoder to serialize all elements"""

    def default(self, o):
        if isinstance(o, CRS):
            return o.to_wkt()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return super().default(o)


def save_disparity_maps(dataset: xr.Dataset, cfg: Dict) -> None:
    """
    Save disparity maps into directory defined by cfg's `output/path` key,
    create it with its parents if necessary.

    :param dataset: Dataset which contains:

        - lines : the disparity map for the lines 2D DataArray (row, col)
        - columns : the disparity map for the columns 2D DataArray (row, col)
    :type dataset: xr.Dataset
    :param cfg: user configuration
    :type cfg: Dict
    :return: None
    """

    # remove ROI margins to save only user ROI in tif files
    if "ROI" in cfg:
        dataset = remove_roi_margins(dataset, cfg)
    if dataset.attrs["transform"] is not None:
        adjust_georeferencement(dataset, cfg)
    # create output dir
    output = Path(cfg["output"]["path"]) / "disparity_map"
    _save_dataset(dataset, output)
    _save_disparity_maps_report(dataset, output)


def _save_disparity_maps_report(dataset: xr.Dataset, output: Path) -> None:
    """
    Generate a report about disparities statistics and save it to json file.
    :param dataset: disparity maps
    :type dataset: xr.Dataset
    :param output: path where to save report
    :type output: Path
    """
    report = {"statistics": {"disparity": reporting.report_disparities(dataset)}}
    with open(output / "report.json", "w", encoding="utf8") as fd:
        json.dump(report, fd, indent=2, cls=AllPrimitiveEncoder)


def _save_dataset(dataset: xr.Dataset, output: Path) -> None:
    """
    Save data_vars in the output directory.

    :param dataset: Dataset
    :type dataset: xr.Dataset
    :param output: output directory
    :type output: Path
    :return: None
    """
    output.mkdir(parents=True, exist_ok=True)

    # dataset["validity"] is the only item to have several bands,
    # so the band_names list will only be used for it.
    for name, data in dataset.items():
        write_data_array(
            data,
            str((output / str(name)).with_suffix(".tif")),
            dtype=data.dtype,
            band_names=list(dataset.criteria.values) if name == "validity" else None,
            crs=dataset.attrs["crs"],
            transform=dataset.attrs["transform"],
        )

    save_attributes(dataset, output)


def save_attributes(dataset: xr.Dataset, output: Union[str, PathLike]) -> None:
    """
    Save dataset attributes in a json file

    :param dataset: Dataset which contains:

        - row_map : the disparity map for the lines 2D DataArray (row, col)
        - col_map : the disparity map for the columns 2D DataArray (row, col)
    :type dataset: xr.Dataset
    :param output: output directory
    :type output: Union[str, PathLike]
    :return: None
    """
    with open(output / Path("attributes.json"), "w", encoding="utf8") as fd:
        json.dump(dataset.attrs, fd, indent=2, cls=AllPrimitiveEncoder)


def adjust_georeferencement(dataset: xr.Dataset, cfg: Dict) -> None:
    """
    Change origin in case a ROI is present and set pixel size to the matching cost step.

    :param dataset: dataset to configure.
    :type dataset: xr.Dataset
    :param cfg: configuration
    :type cfg: Dict
    """
    if "ROI" in cfg:
        # Translate georeferencement origin to ROI origin:
        dataset.attrs["transform"] *= Affine.translation(cfg["ROI"]["col"]["first"], cfg["ROI"]["row"]["first"])
    row_step, col_step = get_step(cfg)
    set_pixel_size(dataset, row_step, col_step)


def get_step(cfg: Dict) -> Tuple[int, int]:
    """
    Get step from matching cost or retun default value.
    :param cfg: configuration
    :type cfg: Dict
    :return: row_step, col_step
    :rtype: Tuple[int, int]
    """
    try:
        return cfg["pipeline"]["matching_cost"]["step"]
    except KeyError:
        return 1, 1


def set_pixel_size(dataset: xr.Dataset, row_step: int = 1, col_step: int = 1) -> None:
    """
    Set the pixel size according to the step used in calculating the matching cost.

    This ensures that all pixels are well geo-referenced in case a step is applied.

    :param dataset: Data to save
    :type dataset: xr.Dataset
    :param row_step: step used in row
    :type row_step: int
    :param col_step: step used in column
    :type col_step: int
    """
    dataset.attrs["transform"] *= Affine.scale(col_step, row_step)


def dataset_disp_maps(
    delta_row: np.ndarray,
    delta_col: np.ndarray,
    coords: Coordinates,
    correlation_score: np.ndarray,
    dataset_validity: xr.Dataset,
    attributes: dict = None,
) -> xr.Dataset:
    """
    Create the dataset containing disparity maps and score maps

    :param delta_row: disparity map for row
    :type delta_row: np.ndarray
    :param delta_col: disparity map for col
    :type delta_col: np.ndarray
    :param coords: disparity maps coordinates
    :type coords: xr.Coordinates
    :param correlation_score: score map
    :type correlation_score: np.ndarray
    :param dataset_validity: xr.Dataset containing validity informations
    :type dataset_validity: xr.Dataset
    :param attributes: disparity map for col
    :type attributes: dict
    :return: dataset: Dataset with the disparity maps and score with the data variables :

            - row_map 2D xarray.DataArray (row, col)
            - col_map 2D xarray.DataArray (row, col)
            - score 2D xarray.DataArray (row, col)
    :rtype: xarray.Dataset
    """

    # Raise an error if col coordinate is missing
    if coords.get("col") is None:
        raise ValueError("The col coordinate does not exist")
    # Raise an error if row coordinate is missing
    if coords.get("row") is None:
        raise ValueError("The row coordinate does not exist")

    coords = {
        "row": coords.get("row"),
        "col": coords.get("col"),
    }

    dims = ("row", "col")

    dataarray_row = xr.DataArray(delta_row, dims=dims, coords=coords)
    dataarray_col = xr.DataArray(delta_col, dims=dims, coords=coords)
    dataarray_score = xr.DataArray(correlation_score, dims=dims, coords=coords)

    dataset = xr.Dataset(
        {
            "row_map": dataarray_row,
            "col_map": dataarray_col,
            "correlation_score": dataarray_score,
        }
    )

    dataset = xr.merge([dataset, dataset_validity])

    if attributes is not None:
        dataset.attrs = attributes

    return dataset


def set_out_of_row_disparity_range_to_other_value(
    data: xr.DataArray,
    min_disp_grid: NDArray[np.floating],
    max_disp_grid: NDArray[np.floating],
    value: Union[int, float, Criteria],
    global_disparity_range: Union[None, List[int]] = None,
) -> None:
    """
    Put special value in data  where the disparity is out of the range defined by disparity grids.

    The operation is done inplace.

    :param data: cost_volumes or criteria_dataarray to modify.
    :type data: xr.DataArray 4D
    :param min_disp_grid: grid of min disparity.
    :type min_disp_grid: NDArray[np.floating]
    :param max_disp_grid: grid of max disparity.
    :type max_disp_grid: NDArray[np.floating]
    :param value: value to set on data.
    :type value: Union[int, float, Criteria]
    :param global_disparity_range:
    :type global_disparity_range:
    """
    ndisp_row = data.shape[-2]

    # We want to put special value on points that are not in the global disparity range (row_disparity_source)
    for disp_row in range(ndisp_row):
        if global_disparity_range is not None:  # Case we are working with cost volume
            masking = np.nonzero(
                np.logical_or(
                    (data.coords["disp_row"].data[disp_row] < min_disp_grid)
                    & (data.coords["disp_row"].data[disp_row] >= global_disparity_range[0]),
                    (data.coords["disp_row"].data[disp_row] > max_disp_grid)
                    & (data.coords["disp_row"].data[disp_row] <= global_disparity_range[1]),
                )
            )
        else:
            masking = np.nonzero(
                np.logical_or(
                    data.coords["disp_row"].data[disp_row] < min_disp_grid,
                    data.coords["disp_row"].data[disp_row] > max_disp_grid,
                )
            )
        data.data[masking[0], masking[1], disp_row, :] = value


def set_out_of_col_disparity_range_to_other_value(
    data: xr.DataArray,
    min_disp_grid: NDArray[np.floating],
    max_disp_grid: NDArray[np.floating],
    value: Union[int, float, Criteria],
    global_disparity_range: Union[None, List[int]] = None,
) -> None:
    """
    Put special value in data (cost_volumes or criteria_dataarray) where the disparity is out of the range defined
    by disparity grids.

    The operation is done inplace.

    :param data: cost_volumes or criteria_dataarray to modify.
    :type data: xr.DataArray 4D
    :param min_disp_grid: grid of min disparity.
    :type min_disp_grid: NDArray[np.floating]
    :param max_disp_grid: grid of max disparity.
    :type max_disp_grid: NDArray[np.floating]
    :param value: value to set on data.
    :type value: Union[int, float, Criteria]
    :param global_disparity_range:
    :type global_disparity_range:
    """
    ndisp_col = data.shape[-1]

    # We want to put special value on points that are not in the global disparity range (col_disparity_source)
    for disp_col in range(ndisp_col):
        if global_disparity_range is not None:  # Case we are working with cost volume
            masking = np.nonzero(
                np.logical_or(
                    (data.coords["disp_col"].data[disp_col] < min_disp_grid)
                    & (data.coords["disp_col"].data[disp_col] >= global_disparity_range[0]),
                    (data.coords["disp_col"].data[disp_col] > max_disp_grid)
                    & (data.coords["disp_col"].data[disp_col] <= global_disparity_range[1]),
                )
            )
        else:
            masking = np.nonzero(
                np.logical_or(
                    data.coords["disp_col"].data[disp_col] < min_disp_grid,
                    data.coords["disp_col"].data[disp_col] > max_disp_grid,
                )
            )
        data.data[masking[0], masking[1], :, disp_col] = value


def save_config(config: Dict) -> None:
    """
    Save config to json file in directory given by the key `output/path`.

    Create file tree if it does not exist,
    :param config: configuration to save
    :type config: Dict
    """
    path_output = Path(config["output"]["path"])
    path_output.mkdir(parents=True, exist_ok=True)
    with open(path_output / "config.json", "w", encoding="utf8") as fd:
        json.dump(config, fd, indent=2, cls=AllPrimitiveEncoder)


def string_to_path(path: str, relative_to: Union[Path, str]) -> Path:
    """
    Get the absolute path of a given path string. If the path is not absolute,
    it resolves it relative to the provided ``relative_to`` path.

    :param path: The path string to convert to an absolute path.
    :type path: str
    :param relative_to: The base path to resolve the relative path.
    :type relative_to: PathLike | str
    :return: The absolute path of the given path string.
    :rtype: Path

    :Example:
        >>> string_to_path('/absolute/path', Path('/home/user'))
        PosixPath('/absolute/path')

        >>> string_to_path('relative/path', Path('/home/user'))
        PosixPath('/home/user/relative/path')

        >>> string_to_path('~/mydir', Path('/home/user'))
        PosixPath('/home/user/mydir')
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def resolve_path_in_config(config: Dict, config_path: Path) -> Dict:
    """
    Create a copy of config with all path strings replaced by an absolute path string relative to
    config_path.

    :param config: config to modify
    :type config: Dict
    :param config_path: path to the config file.
    :type config_path: Path
    :return: The configuration with changed paths.
    :rtype: Dict
    """
    result = deepcopy(config)
    relative_to = config_path.parent
    result["input"]["left"]["img"] = str(string_to_path(config["input"]["left"]["img"], relative_to))
    result["input"]["right"]["img"] = str(string_to_path(config["input"]["right"]["img"], relative_to))
    if not "estimation" in result["pipeline"]:
        col_disparity_init = config["input"]["col_disparity"]["init"]
        if isinstance(col_disparity_init, str):
            result["input"]["col_disparity"]["init"] = str(string_to_path(col_disparity_init, relative_to))
        row_disparity_init = config["input"]["row_disparity"]["init"]
        if isinstance(row_disparity_init, str):
            result["input"]["row_disparity"]["init"] = str(string_to_path(row_disparity_init, relative_to))
    result["output"]["path"] = str(string_to_path(config["output"]["path"], relative_to))
    return result
