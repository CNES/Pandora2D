/* Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA2D
 *
 *     https://github.com/CNES/Pandora2D
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This module contains functions associated to the binding pybind of cpp filter methods.
*/

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "bicubic.hpp"
#include "cardinal_sine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(interpolation_filter_bind, m) {
  py::class_<abstractfilter::AbstractFilter>(m, "AbstractFilter")
      .def_property_readonly("size", &abstractfilter::AbstractFilter::get_size,
                             "Return filter's size.")
      .def("get_coeffs", &abstractfilter::AbstractFilter::get_coeffs, "fractional_shift"_a,
           R"mydelimiter(
            Returns the interpolator coefficients to be applied to the resampling area.

            The size of the returned array depends on the filter margins:
                - For a row shift, returned array size = up_margin + down_margin + 1
                - For a column shift, returned array size = left_margin + right_margin + 1

            :param fractional_shift:
                positive fractional shift of the subpixel position to be interpolated
            :type fractional_shift: float
            :return: a array of interpolator coefficients whose size depends on the filter margins
            :rtype: np.ndarray
            )mydelimiter")
      .def("apply", &abstractfilter::AbstractFilter::apply, "resampling_area"_a, "row_coeff"_a,
           "col_coeff"_a,
           R"mydelimiter(
            Returns the value of the interpolated position

            :param resampling_area: area on which interpolator coefficients will be applied
            :type resampling_area: np.ndarray
            :param row_coeff: interpolator coefficients in rows
            :type row_coeff: np.ndarray
            :param col_coeff: interpolator coefficients in columns
            :type col_coeff: np.ndarray
            :return: the interpolated value of the position corresponding to col_coeff and row_coeff
            :rtype: float
            )mydelimiter")
      .def("interpolate", &abstractfilter::AbstractFilter::interpolate, "image"_a,
           "col_positions"_a, "row_positions"_a, "max_fractional_value"_a,
           R"mydelimiter(
            Returns the values of the 8 interpolated positions

            :param image: image
            :type image: np.ndarray
            :param positions: subpix positions to be interpolated
            :type positions: Tuple[np.ndarray, np.ndarray]
            :param max_fractional_value: maximum fractional value used to get coefficients
            :type max_fractional_value: float
            :return: the interpolated values of the corresponding subpix positions
            :rtype: List of float
            )mydelimiter")
      .def("get_margins", &abstractfilter::AbstractFilter::get_margins,
           "Return margins used by the filter.");

  py::class_<Bicubic, abstractfilter::AbstractFilter>(m, "Bicubic")
      .def(py::init<>())
      .def("get_coeffs", &Bicubic::get_coeffs, "fractional_shift"_a,
           R"mydelimiter(
            Returns the interpolator coefficients to be applied to the resampling area.

            The size of the returned array depends on the filter margins:
                - For a row shift, returned array size = up_margin + down_margin + 1
                - For a column shift, returned array size = left_margin + right_margin + 1

            :param fractional_shift:
                positive fractional shift of the subpixel position to be interpolated.
            :type fractional_shift: float
            :return: an array of interpolator coefficients whose size depends on the filter margins
            :rtype: np.ndarray
            )mydelimiter");

  py::class_<CardinalSine, abstractfilter::AbstractFilter>(m, "CardinalSine",
                                                           R"mydelimiter(
                                                           R"mydelimiter(
            )mydelimiter")
      .def(py::init<const int, const double>(), "half_size"_a = 6, "fractional_shift"_a = 0.25,
           R"mydelimiter(
           R"mydelimiter(
            :type cfg: dict
            :param fractional_shift:
                interval between each interpolated point, sometimes referred to as precision.
                Expected value in the range [0,1[.
            :type fractional_shift: float

            )mydelimiter")
      .def("get_coeffs", &CardinalSine::get_coeffs, "fractional_shift"_a,
           R"mydelimiter(
            Returns the interpolator coefficients to be applied to the resampling area.

            The size of the returned array depends on the filter margins:
                - For a row shift, returned array size = up_margin + down_margin + 1
                - For a column shift, returned array size = left_margin + right_margin + 1

            :param fractional_shift:
                positive fractional shift of the subpixel position to be interpolated.
            :type fractional_shift: float
            :return: a array of interpolator coefficients whose size depends on the filter margins
            :rtype: np.ndarray
            )mydelimiter");

  py::class_<Margins>(m, "Margins")
      .def_readwrite("up", &Margins::up)
      .def_readwrite("down", &Margins::down)
      .def_readwrite("left", &Margins::left)
      .def_readwrite("right", &Margins::right);
}

// Compilation command
// g++ -I eigen-3.4.0 -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes)
// pandora2d/interpolation_filter_cpp/src/bicubic.cpp
// pandora2d/interpolation_filter_cpp/src/interpolation_filter.cpp
// pandora2d/interpolation_filter_cpp/src/bind.cpp
// -o pandora2d/interpolation_filter_cpp/src/filter_bindings_cpp$(python3-config --extension-suffix)