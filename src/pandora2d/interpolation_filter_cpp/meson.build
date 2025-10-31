interpolation_filter_includes = include_directories('include')
interpolation_filter_src = files('src/interpolation_filter.cpp')
cardinal_sine_src = files('src/cardinal_sine.cpp')

bicubic_src = files('src/bicubic.cpp')
bind_src = files('src/bind.cpp')

py.extension_module(
    'interpolation_filter_bind',
    [interpolation_filter_src, bicubic_src, cardinal_sine_src, bind_src],
    subdir: 'pandora2d/interpolation_filter_cpp',
    install: true,
    include_directories: [interpolation_filter_includes, common_includes],
    dependencies: [pybind11_dep, eigen_dep],
)
