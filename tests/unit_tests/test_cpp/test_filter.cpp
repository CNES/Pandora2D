#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest/doctest.h"
#include "../pandora2d/interpolation_filter_cpp/include/interpolation_filter.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace abstractfilter
{

    TEST_SUITE("AbstractFilter apply")
    {

        AbstractFilter filter;

        t_Matrix resampling_area(4, 4);
        t_Vector row_coeff(4);
        t_Vector col_coeff(4);

        TEST_CASE("With identical rows in resampling area")
        {
            resampling_area << 0, 1, 2, 3,
                0, 1, 2, 3,
                0, 1, 2, 3,
                0, 1, 2, 3;

            SUBCASE("0.5 in columns and in rows")
            {
                row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                double expected_result = 1.5;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                CHECK(result == doctest::Approx(expected_result).epsilon(1e-6));
            }

            SUBCASE("0.5 in columns")
            {
                row_coeff << 0.0, 1.0, 0.0, 0.0;
                col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                double expected_result = 1.5;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }

            SUBCASE("0.5 in rows")
            {
                row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                col_coeff << 0.0, 1.0, 0.0, 0.0;
                double expected_result = 1.0;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }

            SUBCASE("0.25 in columns")
            {
                row_coeff << 0.0, 1.0, 0.0, 0.0;
                col_coeff << -0.0703125, 0.8671875, 0.2265625, -0.0234375;
                double expected_result = 1.25;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }

            SUBCASE("0.25 in rows")
            {
                row_coeff << -0.0703125, 0.8671875, 0.2265625, -0.0234375;
                col_coeff << 0.0, 1.0, 0.0, 0.0;
                double expected_result = 1.0;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }
        }

        TEST_CASE("with identical columns in resampling area")
        {
            resampling_area << 0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3;

            SUBCASE("0.5 in columns and in rows")
            {
                row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                double expected_result = 1.5;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }
            SUBCASE("0.5 in columns")
            {
                row_coeff << 0.0, 1.0, 0.0, 0.0;
                col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                double expected_result = 1.0;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }
            SUBCASE("0.5 in rows")
            {
                row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                col_coeff << 0.0, 1.0, 0.0, 0.0;
                double expected_result = 1.5;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }
        }

        TEST_CASE("with 3/4 identical rows in resampling area")
        {

            resampling_area << 0, 1, 2, 3,
                0, 1, 4, 3,
                0, 1, 2, 3,
                0, 1, 2, 3;

            SUBCASE("0.5 in columns")
            {
                row_coeff << 0.0, 1.0, 0.0, 0.0;
                col_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                double expected_result = 2.625;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }

            SUBCASE("0.5 in rows")
            {
                row_coeff << -0.0625, 0.5625, 0.5625, -0.0625;
                col_coeff << 0.0, 1.0, 0.0, 0.0;
                double expected_result = 1.0;
                double result = filter.apply(resampling_area, row_coeff, col_coeff);
                REQUIRE(result == doctest::Approx(expected_result).epsilon(1e-6));
            }
        }
    }
}
