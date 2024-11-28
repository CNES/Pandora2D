/* Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to histogram.
*/

#include "histogram1D.hpp"
#include "bin.hpp"
#include <iostream>



/**
 * @brief Construct a new Histogram 1D
 * 
 * @param _values: on histogram
 * @param _nb_bins: number of bins
 * @param _low_bound: smaller value on histogram
 * @param _bins_width: size of one bin on histogram
 */
Histogram1D::Histogram1D(Eigen::VectorXd &values, std::size_t nb_bins, double low_bound, double bins_width)
    : m_values(values), m_nb_bins(nb_bins), m_low_bound(low_bound), m_bins_width(bins_width)
{}


/**
 * @brief Construct a new Histogram 1D
 * 
 * @param _nb_bins: number of bins
 * @param _low_bound: smaller value on histogram
 * @param _bins_width: size of one bin on histogram
 */
Histogram1D::Histogram1D(std::size_t nb_bins, double low_bound, double bins_width)
    : m_nb_bins(nb_bins), m_low_bound(low_bound), m_bins_width(bins_width)
{
    m_values = Eigen::VectorXd::Zero(m_nb_bins);
}


/**
 * @brief Construct a new Histogram 1D
 * 
 * @param img 
 */
Histogram1D::Histogram1D(const Eigen::MatrixXd &img)
{
    create(img);
    m_values = Eigen::VectorXd::Zero(m_nb_bins);
}


/**
 * @brief Create Histogram 1D object without compute values
 * 
 * @param img 
 */
void Histogram1D::create(const Eigen::MatrixXd &img)
{
    m_bins_width = get_bins_width(img);
    const double dynamique = img.maxCoeff() - img.minCoeff();
    m_nb_bins = static_cast<int>(1.+ (dynamique / m_bins_width));

    // check nb_bins > NB_BINS_MAX
    if (m_nb_bins > NB_BINS_MAX)
        // TO DO update dynamique ici
        m_nb_bins = NB_BINS_MAX;

    m_low_bound = img.minCoeff() - (static_cast<double>(m_nb_bins) * m_bins_width - dynamique)/2.;
}


/**
 * @brief Create and compute Histogram 1D
 * 
 * @param img 
 * @return Histogram1D 
 */
Histogram1D calculate_histogram1D(const Eigen::MatrixXd &img)
{
    auto hist = Histogram1D(img);
    Eigen::VectorXd hist_values = Eigen::VectorXd::Zero(hist.nb_bins());
    auto low_bound = hist.low_bound();
    auto bin_width = hist.bins_width();
    for (auto pixel : img.reshaped())
    {
        auto index = static_cast<int>((pixel - low_bound) / bin_width);
        hist_values[index] += 1;
    }

    hist.set_values(hist_values);
    return hist;
}
