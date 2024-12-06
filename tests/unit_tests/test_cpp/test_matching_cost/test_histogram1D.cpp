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
This module contains tests associated to histogram 1D.
*/


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "histogram1D.hpp"
#include "conftest.hpp"

#include <Eigen/Dense>
#include <iostream>


/**
 * @brief histogram1D calculation medicis version
 * The calculation of the pixel index in the bin can be found here: 
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * quantizer_mgr.c#L305
 * The quantizer params are here:
 * https://gitlab.cnes.fr/OutilsCommuns/medicis/-/blob/master/SOURCES/sources/QPEC/Library/sources/
 * quantizer_mgr.c#L103
 * 
 * @param m : the Eigen matrix
 */
Eigen::VectorXd histo1D_medicis(const Eigen::MatrixXd &m)
{
    // Pandora2D
    auto histo1D = Histogram1D(m);
    std::cout << "PANDORA2D" << std::endl;
    std::cout << "low_bound " << histo1D.low_bound() 
              << " nb_bins " << histo1D.nb_bins() << std::endl;

    // MEDICIS
    auto h0 = histo1D.bins_width();
    auto dynamique = m.maxCoeff() - m.minCoeff();
    auto nb_bins = (int) (1.+ (dynamique/h0));
    auto offset_medicis = (((double) nb_bins)*h0 - dynamique)/2.;
    auto lower_bound_mecidis = m.minCoeff() - offset_medicis;
    auto upper_bound_mecidis = lower_bound_mecidis + h0*((double)nb_bins);
    auto epsilon_medicis = (1.+ (dynamique/h0)) - (double)(nb_bins);
    std::cout << "MEDICIS" << std::endl;
    std::cout << "low_bound " << lower_bound_mecidis << " upper_bound " << upper_bound_mecidis 
        << " nb_bins " << nb_bins << " offset_medicis " << offset_medicis 
        << " epsilon_medicis " << epsilon_medicis << std::endl;


    // offset =  (A_ps_quantizer->lower_bound)*(A_ps_quantizer->quantization_delta ) 
    //           + (A_ps_quantizer->epsilon);
    auto nb_intervals = upper_bound_mecidis; //abs(m.maxCoeff() - m.minCoeff()) + 1;
    auto quantization_delta = ((double) (nb_intervals))/(upper_bound_mecidis-lower_bound_mecidis);
    auto offset = lower_bound_mecidis * quantization_delta + epsilon_medicis;
    std::cout << "nb_intervals " << nb_intervals << " quantization_delta " << quantization_delta 
    << " offset " << offset << std::endl;

    std::cout << "image size " << m.size() << std::endl;
    Eigen::VectorXd indexes_medicis = Eigen::VectorXd::Zero(m.size());
    auto index = indexes_medicis.data();
    for (auto pixel : m.reshaped())
    {
        *index = (int) ((pixel * quantization_delta) - offset );
        std::cout << "pixel value " << pixel << " index " << *index << std::endl;
        index++;
    }
    return indexes_medicis;
}


TEST_CASE("Test constructor")
{

    SUBCASE("With Eigen::VectorXd::Zero")
    {
        Eigen::VectorXd m = Eigen::VectorXd::Zero(3);
        Histogram1D hist = Histogram1D(m);

        check_inside_eigen_element(hist.values(), Eigen::VectorXd::Zero(1));
        CHECK(hist.nb_bins() == 1);
        CHECK(hist.low_bound() == -0.5);
        CHECK(hist.bins_width() == 1);
    }

    SUBCASE("With Eigen::VectorXd {1,2,3,4}")
    {
        Eigen::VectorXd m(4);
        m << 1, 2, 3, 4;

        Histogram1D hist = Histogram1D(m);

        check_inside_eigen_element(hist.values(), Eigen::VectorXd::Zero(2));
        CHECK(hist.nb_bins() == 2);
        CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
    }

    SUBCASE("First constructor")
    {
        Eigen::VectorXd m = Eigen::VectorXd::Ones(3);
        Histogram1D hist = Histogram1D(m, 3, 0.1, 1.3);
        
        check_inside_eigen_element(hist.values(), m);
        CHECK(hist.nb_bins() == 3);
        CHECK(hist.low_bound() == 0.1);
        CHECK(hist.bins_width() == 1.3);
    }

    SUBCASE("Second constructor")
    {
        Histogram1D hist = Histogram1D(2, 0.1, 1.3);
        
        check_inside_eigen_element(hist.values(), Eigen::VectorXd::Zero(2));
        CHECK(hist.nb_bins() == 2);
        CHECK(hist.low_bound() == 0.1);
        CHECK(hist.bins_width() == 1.3);
    }
}


TEST_CASE("Test calculate_histogram1D function")
{
    SUBCASE("positive low_bound & matrix coefficients")
    {
        Eigen::MatrixXd m(1, 4);
        m << 1, 2, 3, 4;

        auto hist = calculate_histogram1D(m);

        check_inside_eigen_element(hist.values(), Eigen::VectorXd::Ones(2)*2);
        CHECK(hist.nb_bins() == 2);
        CHECK(hist.low_bound() == doctest::Approx(0.0412283).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
    }

    SUBCASE("negative low_bound & positive matrix coefficients")
    {
        Eigen::MatrixXd m(4, 4);
        m << 1.0,   2.0,  3.0,  4.0,
             5.0,   6.0,  7.0,  8.0,
             9.0,  10.0, 11.0, 12.0,
             13.0, 14.0, 15.0, 16.0;
        
        Eigen::MatrixXd hist_expected(3, 1);
        hist_expected << 5, 6, 5;

        auto hist = calculate_histogram1D(m);

        check_inside_eigen_element(hist.values(), hist_expected);
        CHECK(hist.nb_bins() == 3);
        CHECK(hist.low_bound() == doctest::Approx(-1.0795972).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(6.3863981).epsilon(1e-7));
    }

    SUBCASE("negative low_bound & matrix coefficients")
    {
        Eigen::MatrixXd m(1, 4);
        m << -11, -12, -13, -14;

        auto hist = calculate_histogram1D(m);

        check_inside_eigen_element(hist.values(), Eigen::VectorXd::Ones(2)*2);
        CHECK(hist.nb_bins() == 2);
        CHECK(hist.low_bound() == doctest::Approx(-14.9587716).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(2.4587717).epsilon(1e-7));
    }

    SUBCASE("positive & negative matrix coefficients")
    {
        Eigen::MatrixXd m(4, 4);
        m << -0.1,  -0.2,    0.30,    0.40,
              0.1,   0.3,   -0.45,   -0.59,
              0.99, -0.101,  0.11452, 0.1235,
             -0.36, -0.256, -0.56,   -0.1598;
        
        Eigen::MatrixXd hist_expected(3, 1);
        hist_expected << 9, 6, 1;

        auto hist = calculate_histogram1D(m);

        check_inside_eigen_element(hist.values(), hist_expected);
        CHECK(hist.nb_bins() == 3);
        CHECK(hist.low_bound() == doctest::Approx(-0.6199559).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(0.5466373).epsilon(1e-7));
    }

    SUBCASE("compare with Medicis")
    {
        Eigen::MatrixXd m(5, 5);
        m << 1600.0, 1695.0, 1630.0, 1564.0, 1684.0,
             1523.0, 1480.0, 1424.0, 1401.0, 1407.0,
             1480.0, 1407.0, 1382.0, 1419.0, 1425.0,
             1445.0, 1368.0, 1368.0, 1526.0, 1574.0,
             1409.0, 1340.0, 1399.0, 1578.0, 1644.0;
        
        Eigen::MatrixXd hist_expected(3, 1);
        hist_expected <<13, 7, 5;

        auto hist = calculate_histogram1D(m);

        check_inside_eigen_element(hist.values(), hist_expected);
        CHECK(hist.nb_bins() == 3);
        CHECK(hist.low_bound() == doctest::Approx(1330.078490).epsilon(1e-7));
        CHECK(hist.bins_width() == doctest::Approx(124.947673).epsilon(1e-7));
        /// 1330 -> 1454 -> 1578 -> 1702
    }

    // SUBCASE("with nan value")
    // {
    //     Eigen::MatrixXd m(4, 4);
    //     m << -0.1, -0.2, 0.30, std::numeric_limits<double>::quiet_NaN(),
    //          0.1, 0.3, -0.45, -0.59,
    //          0.99, -0.101, 0.11452, 0.1235,
    //          -0.36, -0.256, -0.56, -0.1598;
        
    //     Eigen::MatrixXd hist_expected(3, 1);
    //     hist_expected << 9, 6, 1;

    //     auto hist = calculate_histogram1D(m);

    //     check_inside_eigen_element(hist.values(), hist_expected);
    //     CHECK(hist.nb_bins() == 3);
    //     CHECK(hist.low_bound() == doctest::Approx(-0.6199559).epsilon(1e-7));
    //     CHECK(hist.bins_width() == doctest::Approx(0.5466373).epsilon(1e-7));
    // }
}
