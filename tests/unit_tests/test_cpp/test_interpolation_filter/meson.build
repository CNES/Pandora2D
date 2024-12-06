test(
    'Filter',
    executable(
        'test_filter',
        ['test_filter.cpp', interpolation_filter_src, bicubic_src],
        include_directories: interpolation_filter_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Bicubic',
    executable(
        'test_bicubic',
        ['test_bicubic.cpp', interpolation_filter_src, bicubic_src],
        include_directories: interpolation_filter_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)