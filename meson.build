project(
  'pandora2d',
  'cpp',
  version: run_command(
    'version.sh',
    'get-vcs',
  ).stdout().strip(),
  default_options: [
    'cpp_std=c++17',
  ],
)

meson.add_dist_script('version.sh', 'set-dist', meson.project_version())

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
subdir('pandora2d/matching_cost_cpp')
subdir('tests/unit_tests/test_cpp')
subdir('tests/unit_tests/test_cpp/test_matching_cost')
