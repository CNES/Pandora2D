test(
    'Histogram1D',
    executable(
        'test_histogram1D',
        ['test_histogram1D.cpp', matching_cost_src],
        include_directories:matching_cost_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Histogram2D',
    executable(
        'test_histogram2D',
        ['test_histogram2D.cpp', matching_cost_src],
        include_directories:matching_cost_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Bin',
    executable(
        'test_bin',
        ['test_bin.cpp'],
        include_directories:matching_cost_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Operation',
    executable(
        'test_operation',
        ['test_operation.cpp'],
        include_directories:matching_cost_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)