test(
    'Dichotomy',
    executable(
        'test_dichotomy',
        [
            'test_dichotomy.cpp',
            refinement_src,
            interpolation_filter_src,
            bicubic_src,
            test_src,
            common_src,
        ],
        include_directories: [
            refinement_includes,
            interpolation_filter_includes,
            test_includes,
            common_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)
