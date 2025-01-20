matching_cost_includes = include_directories('include')
matching_cost_src = files('src/histogram1D.cpp', 'src/histogram2D.cpp', 'src/operation.cpp', 'src/bin.cpp', 
                            'src/mutual_information.cpp', 'src/cost_volumes.cpp')

bind_src = files('src/bind.cpp')

py.extension_module(
    'matching_cost_bind',
    [matching_cost_src, bind_src],
    subdir: 'pandora2d/matching_cost_cpp',
    install: true,
    include_directories : matching_cost_includes,
    dependencies : [pybind11_dep, eigen_dep],
)
