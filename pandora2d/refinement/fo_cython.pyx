#!/usr/bin/env python
# coding: utf8
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
This module contains functions associated to the optical flow method used in the refinement step.
"""

import numpy as np
cimport numpy as np
import itertools
import cython

np.import_array()

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def optical_flow(np.ndarray[float, ndim = 2] grad_x,
                np.ndarray[float, ndim = 2] grad_y,
                np.ndarray[float, ndim = 2] grad_t,
                int w,
                np.ndarray[float, ndim = 2] left,
                float invalid_disp,
                np.ndarray[float, ndim = 2] dec_row,
                np.ndarray[float, ndim = 2] dec_col):

    cdef int i, j
    cdef int l_shape_0 = left.shape[0]
    cdef int l_shape_1 = left.shape[1]
    cdef np.ndarray A = np.zeros([w ** 2, 2], dtype=np.float)
    cdef np.ndarray B = np.zeros([w ** 2, 1], dtype=np.float)
    cdef(DTYPE_t, DTYPE_t) motion

    for ind_row, ind_col in itertools.product(range(w, l_shape_0 - w), range(w, l_shape_1 - w)):

        # Select pixel and neighbourhoods
        Ix = grad_x[ind_row - w: ind_row + w + 1, ind_col - w: ind_col + w + 1]
        Iy = grad_y[ind_row - w: ind_row + w + 1, ind_col - w: ind_col + w + 1]
        It = grad_t[ind_row - w: ind_row + w + 1, ind_col - w: ind_col + w + 1]

        # Create A et B matrix for Lucas Kanade
        A = np.vstack((Ix.flatten(), Iy.flatten())).T
        B = np.reshape(It, len(It) ** 2)[np.newaxis].T

        # v = (A^T.A)^-1.A^T.B
        try:
            motion = tuple(np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B)))
        # if matrix is full of NaN or 0
        except np.linalg.LinAlgError:
            motion = [invalid_disp, invalid_disp]

        dec_row[ind_row, ind_col] = dec_row[ind_row, ind_col] + motion[1]
        dec_col[ind_row, ind_col] = dec_col[ind_row, ind_col] + motion[0]

    return dec_row, dec_col