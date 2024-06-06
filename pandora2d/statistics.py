#  Copyright (c) 2024. Centre National d'Etudes Spatiales (CNES).
#
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

"""Functions to compute statistics on results."""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import xarray as xr


@dataclass
class Statistics:
    """Group various statistics."""

    mean: Union[float, np.floating]
    std: Union[float, np.floating]

    def __str__(self):
        return f"Mean: {self.mean} ± {self.std}"

    def to_dict(self) -> Dict:
        """Convert statistics into a dictionary."""
        # We need to cast to float because np.float can not be serialized to JSON
        return {"mean": float(self.mean), "std": float(self.std)}


def compute_statistics(dataarray: xr.DataArray, invalid_values: Union[np.floating, np.integer] = None) -> Statistics:
    """Compute statistics of a dataArray.

    :param dataarray: data to compute statistics from.
    :type dataarray: xr.DataArray
    :param invalid_values: value to exclude from computation
    :type invalid_values: Union[np.floating, np.integer]
    :return: computed statistics
    :rtype: Statistics
    """
    if invalid_values is None:
        data = dataarray.to_numpy()
    else:
        mask = np.isin(dataarray.to_numpy(), invalid_values, invert=True)
        data = dataarray.to_numpy()[mask]
    return Statistics(mean=np.nanmean(data), std=np.nanstd(data))
