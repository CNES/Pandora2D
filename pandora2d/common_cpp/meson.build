common_includes = include_directories('include')
common_src = files('src/operation.cpp')

bind_src = files('src/bind.cpp')

py.extension_module(
    'common_bind',
    bind_src,
    subdir: 'pandora2d/common_cpp',
    install: true,
    include_directories: common_includes,
    dependencies: [pybind11_dep, eigen_dep],
)
