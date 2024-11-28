interpolation_filter_includes = include_directories('include')
interpolation_filter_src = files('src/interpolation_filter.cpp')

bicubic_src = files('src/bicubic.cpp')
bind_src = files('src/bind.cpp')

py.extension_module(
    'interpolation_filter_bind',
    [interpolation_filter_src, bicubic_src, bind_src],
    install: true,
    include_directories: interpolation_filter_includes,
    dependencies: [pybind11_dep, eigen_dep, doctest_dep],
)
