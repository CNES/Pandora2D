project(
  'pandora2d',
  'cpp',
  version: '0.4.0',
  default_options: [
    'cpp_std=c++17',
  ],
)

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')
eigen_dep = dependency('eigen3')
doctest_dep = dependency('doctest')

install_subdir(
  'pandora2d',
  install_dir: py.get_install_dir() / 'pandora2d',
  strip_directory: true,
)

subdir('pandora2d/interpolation_filter_cpp')
subdir('tests/unit_tests/test_cpp')