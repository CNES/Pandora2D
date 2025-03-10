test(
    'Histogram1D',
    executable(
        'test_histogram1D',
        ['test_histogram1D.cpp', matching_cost_src, test_src],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
        ],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Histogram2D',
    executable(
        'test_histogram2D',
        ['test_histogram2D.cpp', matching_cost_src, test_src],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
        ],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Bin',
    executable(
        'test_bin',
        ['test_bin.cpp', matching_cost_src],
        include_directories: [matching_cost_includes, common_includes],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Operation',
    executable(
        'test_operation',
        ['test_operation.cpp', matching_cost_src],
        include_directories: [matching_cost_includes, common_includes],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'MutualInformation',
    executable(
        'test_mutual_information',
        ['test_mutual_information.cpp', matching_cost_src],
        include_directories: [matching_cost_includes, common_includes],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'CostVolumes',
    executable(
        'test_cost_volumes',
        ['test_cost_volumes.cpp', matching_cost_src, test_src],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
        ],
        dependencies: [eigen_dep, doctest_dep],
    ),
)
