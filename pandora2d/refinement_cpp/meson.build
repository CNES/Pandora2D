refinement_includes = include_directories('include')
refinement_src = files('src/dichotomy.cpp')

py.extension_module(
    'refinement_cpp',
    [refinement_src, interpolation_filter_src, bicubic_src],
    subdir: 'pandora2d',
    install: true,
    include_directories : [refinement_includes, interpolation_filter_includes],
    dependencies : [pybind11_dep, eigen_dep],
)