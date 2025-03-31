project(
    'pandora2d',
    'cpp',
    version: run_command('version.sh', 'get-vcs').stdout().strip(),
    default_options: ['cpp_std=c++17'],
    meson_version: '>=1.1',
)

meson.add_dist_script('version.sh', 'set-dist', meson.project_version())

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')
eigen_dep = dependency('eigen3')
doctest_dep = dependency('doctest', required: get_option('build_cpp_tests'))
python_dep = py.dependency(embed: true, required: get_option('build_cpp_tests'))

install_subdir(
    'pandora2d',
    install_dir: py.get_install_dir() / 'pandora2d',
    strip_directory: true,
)

subdir('pandora2d/common_cpp')
subdir('pandora2d/interpolation_filter_cpp')
subdir('pandora2d/matching_cost_cpp')
subdir('pandora2d/refinement_cpp')

if get_option('build_cpp_tests').enabled()
    subdir('tests/unit_tests/test_cpp')
endif
