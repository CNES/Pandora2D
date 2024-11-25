matching_cost_includes = include_directories('include')
matching_cost_src = files('src/histogram1D.cpp', 'src/histogram2D.cpp', 'src/operation.cpp', 'src/bin.cpp')

py.extension_module(
    'matching_cost_cpp',
    matching_cost_src,
    subdir: 'pandora2d',
    install: true,
    include_directories : matching_cost_includes,
    dependencies : [pybind11_dep, eigen_dep],
)
