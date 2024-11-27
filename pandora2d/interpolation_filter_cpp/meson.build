interpolation_filter_includes = include_directories('include')
interpolation_filter_src = files('src/interpolation_filter.cpp')

py.extension_module(
    'interpolation_filter_cpp',
    interpolation_filter_src,
    subdir: 'pandora2d',
    install: true,
    include_directories: interpolation_filter_includes,
    dependencies: [pybind11_dep, eigen_dep, doctest_dep],
)

bicubic_src = files('src/bicubic.cpp')

py.extension_module(
    'bicubic_cpp',
    bicubic_src,
    subdir: 'pandora2d',
    install: true,
    include_directories: interpolation_filter_includes,
    dependencies: [pybind11_dep, eigen_dep, doctest_dep],
)