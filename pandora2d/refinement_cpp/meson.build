refinement_includes = include_directories('include')
refinement_src = files('src/dichotomy.cpp')

bind_src = files('src/bind.cpp')

py.extension_module(
    'refinement_bind',
    [refinement_src, interpolation_filter_src, bind_src, common_src],
    subdir: 'pandora2d/refinement_cpp',
    install: true,
    include_directories: [
        refinement_includes,
        interpolation_filter_includes,
        common_includes,
    ],
    dependencies: [pybind11_dep, eigen_dep],
)
