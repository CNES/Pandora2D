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
This module contains functions associated to the histogram for cpp.
*/

#ifndef HISTOGRAM1D_HPP
#define HISTOGRAM1D_HPP

#include <Eigen/Dense>

/**
 * @brief Instanciation of Histogram on 1D (one dimension)
 * 
 */
class Histogram1D
{

public:

    /**
     * @brief Construct a new Histogram1D object
     * @param values: on histogram
     * @param nb_bins: number of bins
     * @param low_bound: smaller value on histogram
     * @param bins_width: size of one bin on histogram
     */
    Histogram1D(Eigen::VectorXd &values, std::size_t nb_bins, double low_bound, double bins_width);

    /**
     * @brief Construct a new Histogram1D object
     * @param nb_bins: number of bins
     * @param low_bound: smaller value on histogram
     * @param bins_width: size of one bin on histogram
     * 
     */
    Histogram1D(std::size_t nb_bins, double low_bound, double bins_width);

    /**
     * @brief Construct a new Histogram1D object
     * @param img : image
     * 
     */
    Histogram1D(const Eigen::MatrixXd &img);
    
    /**
     * @brief Destroy the Histogram1D object
     * 
     */
    ~Histogram1D() = default;

    /**
     * @brief Get the Values object
     * 
     * @return const Eigen::VectorXd& 
     */
    const Eigen::VectorXd& getValues() const { return m_values; };

    /**
     * @brief Get the Nb Bins object
     * 
     * @return std::size_t 
     */
    std::size_t getNbBins() const { return m_nb_bins; };

    /**
     * @brief Get the Low Bound object
     * 
     * @return double 
     */
    double getLowBound() const { return m_low_bound; };

    /**
     * @brief Get the Bin Width object
     * 
     * @return double 
     */
    double getBinWidth() const { return m_bins_width; };

    /**
     * @brief Set the Values object
     * 
     * @param values 
     */
    void setValues(const Eigen::VectorXd& values) { m_values = values; };


private:

    void create(const Eigen::MatrixXd &img);

    Eigen::VectorXd m_values; ///< values on histogram
    std::size_t m_nb_bins;    ///< number of bins
    double m_low_bound;       ///< smaller value on histogram
    double m_bins_width;      ///< size of one bin
};

/**
 * @brief Calculate histogram 1D based on an image
 * 
 * @param img 
 * @return Histogram1D 
 */
Histogram1D calculate_histogram1D(const Eigen::MatrixXd &img);

#endif