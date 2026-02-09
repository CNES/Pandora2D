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
import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import rasterio
import xarray as xr
from pandora.common import write_data_array
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning

from pandora2d import reporting

# mypy: disable-error-code="attr-defined, no-redef"
# pylint: disable=useless-import-alias

# xarray.Coordinates corresponds to the latest version of xarray.
# xarray.Coordinate corresponds to the version installed by the artifactory.
# Try/except block to be deleted once the version of xarray has been updated by CNES.
try:
    from xarray import Coordinates as Coordinates
except ImportError:
    from xarray import Coordinate as Coordinates

# Filter Warning when writing not georeferenced image with Rasterio
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

T = TypeVar("T")


class Registry(Generic[T]):
    """Registry of classes.

    A class to decorate classes in order to register them with a string name.
    """

    def __init__(self, default: type[T] | None = None) -> None:
        """
        Initialize the registry with an optional default class.

        :param default: Default class to return if name is not registered. If None, will raise a KeyError.
        """
        self.registered: dict[str, type[T]] = {}
        self.default = default

    def add(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Register a class with `name`.

        :param name: Name to register the decorated class with.
        :return: The decorated class.
        """

        def decorator(cls: type[T]) -> type[T]:
            """Returned decorator used for register class."""
            # No verification is made about already registered name: we need to decide on desired behavior
            self.registered[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """
        Get the class registered as `name`.

        :param name: The name of the registered class to retrieve.
        :return: The class registered under `name` or the default class if not found.
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


def convert_disp_to_grid(dataset: xr.Dataset, pixel_convention: list[int]) -> xr.Dataset:
    """
    Convet disparity maps to deformation grids

    :param dataset: disparity maps dataset
    :param pixel_convention: initial pixel convention for grid
    """

    dataset = dataset.rename_vars({"row_map": "row_deformation_map", "col_map": "col_deformation_map"})

    col_coords_2d, row_coords_2d = np.meshgrid(dataset["col"].values, dataset["row"].values)
    dataset["row_deformation_map"].data += row_coords_2d + pixel_convention[0]
    dataset["col_deformation_map"].data += col_coords_2d + pixel_convention[1]

    return dataset


def convert_grid_to_disp(dataset: xr.Dataset, pixel_convention: list[int]) -> xr.Dataset:
    """
    Convet deformation grids to disparity maps

    :param dataset: deformation maps dataset
    :param pixel_convention: initial pixel convention for grid
    """

    dataset = dataset.rename_vars({"row_deformation_map": "row_map", "col_deformation_map": "col_map"})

    col_coords_2d, row_coords_2d = np.meshgrid(dataset["col"].values, dataset["row"].values)

    dataset["row_map"].data -= row_coords_2d + pixel_convention[0]
    dataset["col_map"].data -= col_coords_2d + pixel_convention[1]

    return dataset


def save_disparity_maps(dataset: xr.Dataset, cfg: dict) -> None:
    """
    Save disparity maps into directory defined by cfg's `output/path` key,
    create it with its parents if necessary.

    :param dataset: Dataset which contains:

        - lines : the disparity map for the lines 2D DataArray (row, col)
        - columns : the disparity map for the columns 2D DataArray (row, col)
    :param cfg: user configuration
    :return: None
    """

    if dataset.attrs["transform"] is not None:
        adjust_georeferencement(dataset, cfg)

    if "deformation_grid" in cfg["output"]:
        dataset = convert_disp_to_grid(dataset, cfg["output"]["deformation_grid"]["init_pixel_conv_grid"])

    output = Path(cfg["output"]["path"]) / "disparity_map"

    if "confidence_measure" in dataset.data_vars:
        save_confidence_maps(dataset, cfg)
        _save_dataset(dataset.drop_vars("confidence_measure"), output)
    else:
        _save_dataset(dataset, output)

    _save_disparity_maps_report(dataset, output)


def save_confidence_maps(dataset: xr.Dataset, cfg: dict) -> None:
    """
    Save confidence maps into directory defined by cfg's `output/path` key,
    create it with its parents if necessary.

    :param dataset: Dataset which contains:

        - lines : the confidence map for the lines 2D DataArray (row, col)
        - columns : the confidence map for the columns 2D DataArray (row, col)
    :param cfg: user configuration
    :return: None
    """

    output = Path(cfg["output"]["path"]) / "cost_volumes"

    output.mkdir(parents=True, exist_ok=True)

    write_data_array(
        dataset["confidence_measure"],
        str((output / "confidence_measure").with_suffix(".tif")),
        dtype=dataset["confidence_measure"].dtype,
        band_names=None,
        crs=dataset.attrs["crs"],
        transform=dataset.attrs["transform"],
    )


def _save_disparity_maps_report(dataset: xr.Dataset, output: Path) -> None:
    """
    Generate a report about disparities statistics and save it to json file.
    :param dataset: disparity maps
    :param output: path where to save report
    """
    report = {"statistics": {"disparity": reporting.report_disparities(dataset)}}
    with open(output / "report.json", "w", encoding="utf8") as fd:
        json.dump(report, fd, indent=2, cls=AllPrimitiveEncoder)


def _save_dataset(dataset: xr.Dataset, output: Path) -> None:
    """
    Save data_vars in the output directory.

    :param dataset: Dataset
    :param output: output directory
    :return: None
    """
    output.mkdir(parents=True, exist_ok=True)

    for name, data_array in dataset.items():
        # Rasterio expects a 3D array with bands on the first axis, while in the dataset
        # bands are on the third axis (when present). We therefore move the third axis
        # to the first position. If data_array was 2D, we add a singleton axis first.
        data = np.moveaxis(np.atleast_3d(data_array.data), -1, 0)
        count, row, col = data.shape

        nodata = dataset.attrs["invalid_disp"] if name in ("row_map", "col_map", "correlation_score") else None

        if name == "validity":
            band_descriptions = list(dataset.criteria.values)
            with rasterio.open(
                (output / str(name)).with_suffix(".tif"),
                mode="w+",
                nbits=1,
                driver="GTiff",
                width=col,
                height=row,
                count=count,
                dtype=data.dtype,
                crs=dataset.attrs["crs"],
                transform=dataset.attrs["transform"],
                nodata=nodata,
            ) as source_ds:
                source_ds.write(data)
                source_ds.descriptions = band_descriptions
        else:
            band_descriptions = [name]

            with rasterio.open(
                (output / str(name)).with_suffix(".tif"),
                mode="w+",
                driver="GTiff",
                width=col,
                height=row,
                count=count,
                dtype=data.dtype,
                crs=dataset.attrs["crs"],
                transform=dataset.attrs["transform"],
                nodata=nodata,
            ) as source_ds:
                source_ds.write(data)
                source_ds.descriptions = band_descriptions

    save_attributes(dataset, output)


def save_attributes(dataset: xr.Dataset, output: str | PathLike) -> None:
    """
    Save dataset attributes in a json file

    :param dataset: Dataset which contains:

        - row_map : the disparity map for the lines 2D DataArray (row, col)
        - col_map : the disparity map for the columns 2D DataArray (row, col)
    :param output: output directory
    :return: None
    """
    with open(output / Path("attributes.json"), "w", encoding="utf8") as fd:
        json.dump(dataset.attrs, fd, indent=2, cls=AllPrimitiveEncoder)


def adjust_georeferencement(dataset: xr.Dataset, cfg: dict) -> None:
    """
    Change origin in case a ROI is present and set pixel size to the matching cost step.

    :param dataset: dataset to configure.
    :param cfg: configuration
    """
    if "ROI" in cfg:
        # Translate georeferencement origin to ROI origin:
        dataset.attrs["transform"] *= Affine.translation(cfg["ROI"]["col"]["first"], cfg["ROI"]["row"]["first"])
    row_step, col_step = get_step(cfg)
    set_pixel_size(dataset, row_step, col_step)


def get_step(cfg: dict) -> tuple[int, int]:
    """
    Get step from matching cost or retun default value.
    :param cfg: configuration
    :return: row_step, col_step
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
    :param row_step: step used in row
    :param col_step: step used in column
    """
    dataset.attrs["transform"] *= Affine.scale(col_step, row_step)


def dataset_disp_maps(
    coords: Coordinates,
    dataset_validity: xr.Dataset,
    attributes: dict = None,
    dtype: np.typing.DTypeLike = np.float32,
) -> xr.Dataset:
    """
    Create the dataset containing disparity maps and score maps
    :param coords: disparity maps coordinates
    :param dataset_validity: xr.Dataset containing validity informations
    :param attributes: disparity map for col
    :param dtype: dtype of the dataset
    :return: dataset: Dataset with the empty disparity maps and score with the data variables :

            - row_map 2D xarray.DataArray (row, col)
            - col_map 2D xarray.DataArray (row, col)
            - score 2D xarray.DataArray (row, col)

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
    shape = (len(coords.get("row")), len(coords.get("col")))

    dataset = xr.Dataset(
        {
            "row_map": (dims, np.full(shape, attributes["invalid_disp"], dtype=dtype)),
            "col_map": (dims, np.full(shape, attributes["invalid_disp"], dtype=dtype)),
            "correlation_score": (dims, np.full(shape, attributes["invalid_disp"], dtype=dtype)),
        },
        coords=coords,
    )

    dataset = xr.merge([dataset, dataset_validity])

    if attributes is not None:
        dataset.attrs = attributes

    return dataset


def fill_dataset_disp_maps(
    disparity_dataset: xr.Dataset,
    delta_row: np.ndarray,
    delta_col: np.ndarray,
    correlation_score: np.ndarray,
) -> None:
    """
    Fill the dataset with computed disparity maps and score maps

    :param disparity_dataset: initialized disparity maps dataset
    :param delta_row: disparity map for row
    :param delta_col: disparity map for col
    :param correlation_score: score map
    """

    disparity_dataset["row_map"].data = delta_row
    disparity_dataset["col_map"].data = delta_col
    disparity_dataset["correlation_score"].data = correlation_score


def save_config(config: dict) -> None:
    """
    Save config to json file in directory given by the key `output/path`.

    Create file tree if it does not exist,
    :param config: configuration to save
    """
    path_output = Path(config["output"]["path"])
    path_output.mkdir(parents=True, exist_ok=True)
    with open(path_output / "config.json", "w", encoding="utf8") as fd:
        json.dump(config, fd, indent=2, cls=AllPrimitiveEncoder)


def string_to_path(path: str, relative_to: Path | str) -> Path:
    """
    Get the absolute path of a given path string. If the path is not absolute,
    it resolves it relative to the provided ``relative_to`` path.

    :param path: The path string to convert to an absolute path.
    :param relative_to: The base path to resolve the relative path.
    :return: The absolute path of the given path string.

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


def resolve_path_in_config(config: dict, config_path: Path) -> dict:
    """
    Create a copy of config with all path strings replaced by an absolute path string relative to
    config_path.

    :param config: config to modify
    :param config_path: path to the config file.
    :return: The configuration with changed paths.
    """
    result = deepcopy(config)
    relative_to = config_path.parent
    result["input"]["left"]["img"] = str(string_to_path(config["input"]["left"]["img"], relative_to))
    result["input"]["right"]["img"] = str(string_to_path(config["input"]["right"]["img"], relative_to))

    if left_mask := config["input"]["left"].get("mask"):
        result["input"]["left"]["mask"] = str(string_to_path(left_mask, relative_to))
    if right_mask := config["input"]["right"].get("mask"):
        result["input"]["right"]["mask"] = str(string_to_path(right_mask, relative_to))

    if "estimation" not in result["pipeline"]:
        col_disparity_init = config["input"]["col_disparity"]["init"]
        if isinstance(col_disparity_init, str):
            result["input"]["col_disparity"]["init"] = str(string_to_path(col_disparity_init, relative_to))
        row_disparity_init = config["input"]["row_disparity"]["init"]
        if isinstance(row_disparity_init, str):
            result["input"]["row_disparity"]["init"] = str(string_to_path(row_disparity_init, relative_to))
    result["output"]["path"] = str(string_to_path(config["output"]["path"], relative_to))
    return result


def all_same(iterable: Iterable) -> bool:
    """Return True if all items in sequence are equals."""
    return len(set(iterable)) == 1
