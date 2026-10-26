test(
    'Filter',
    executable(
        'test_filter',
        ['test_filter.cpp', interpolation_filter_src, bicubic_src, test_src],
        include_directories: [
            interpolation_filter_includes,
            common_includes,
            test_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Bicubic',
    executable(
        'test_bicubic',
        ['test_bicubic.cpp', interpolation_filter_src, bicubic_src],
        include_directories: [interpolation_filter_includes, common_includes],
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Cardinal Sine',
    executable(
        'test_cardinal_sine',
        ['test_cardinal_sine.cpp', interpolation_filter_src, cardinal_sine_src],
        include_directories: [interpolation_filter_includes, common_includes],
        dependencies: [eigen_dep, doctest_dep],
    ),
)
