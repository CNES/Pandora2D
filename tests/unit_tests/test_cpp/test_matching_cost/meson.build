conftest_includes = include_directories('.')
conftest_src = files('conftest.cpp')

test(
    'Histogram1D',
    executable(
        'test_histogram1D',
        [
            'test_histogram1D.cpp',
            matching_cost_src,
            test_src,
            common_src,
            conftest_src,
        ],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
            conftest_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Histogram2D',
    executable(
        'test_histogram2D',
        [
            'test_histogram2D.cpp',
            matching_cost_src,
            test_src,
            common_src,
            conftest_src,
        ],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
            conftest_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Bin',
    executable(
        'test_bin',
        ['test_bin.cpp', matching_cost_src, common_src],
        include_directories: [matching_cost_includes, common_includes],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Mutual Information',
    executable(
        'test_mutual_information',
        ['test_mutual_information.cpp', matching_cost_src, common_src],
        include_directories: [matching_cost_includes, common_includes],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'ZNCC',
    executable(
        'test_zncc',
        ['test_zncc.cpp', matching_cost_src, test_src, common_src],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Compute Cost Volumes',
    executable(
        'test_compute_cost_volumes',
        [
            'test_compute_cost_volumes.cpp',
            matching_cost_src,
            test_src,
            common_src,
            conftest_src,
        ],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
            conftest_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
)

test(
    'Compute Cost Volumes Mutual Information',
    executable(
        'test_compute_cost_volumes_mi',
        [
            'test_compute_cost_volumes_mi.cpp',
            matching_cost_src,
            test_src,
            common_src,
            conftest_src,
        ],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
            conftest_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
    env: ['DATA_PATH=' + meson.current_source_dir()],
)

test(
    'Compute Cost Volumes ZNCC',
    executable(
        'test_compute_cost_volumes_zncc',
        [
            'test_compute_cost_volumes_zncc.cpp',
            matching_cost_src,
            test_src,
            common_src,
            conftest_src,
        ],
        include_directories: [
            matching_cost_includes,
            test_includes,
            common_includes,
            conftest_includes,
        ],
        dependencies: [python_dep, pybind11_dep, eigen_dep, doctest_dep],
    ),
    env: ['DATA_PATH=' + meson.current_source_dir()],
)

