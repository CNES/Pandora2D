test(
    'Operation',
    executable(
        'test_operation',
        ['test_operation.cpp', common_src],
        include_directories: common_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)

test(
    'Cost volume struct',
    executable(
        'test_cost_volume',
        'test_cost_volume.cpp',
        include_directories: common_includes,
        dependencies: [eigen_dep, doctest_dep],
    ),
)
