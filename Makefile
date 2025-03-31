# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
# Dependencies : python3 venv internal module
# Recall: .PHONY  defines special targets not associated with files
#
# Some Makefile global variables can be set in make command line:
# PANDORA2D_VENV: Change directory of installed venv (default local "venv" dir)
#

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: PANDORA2D_VENV="other-venv/" make install
ifndef PANDORA2D_VENV
	PANDORA2D_VENV = "venv"
endif

# Check python3 globally
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version supported globally
PYTHON_VERSION_MIN = 3.9
PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')
ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif


################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      PANDORA2D MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: venv
venv: ## create virtualenv in PANDORA2D_VENV directory if not exists
	@test -d ${PANDORA2D_VENV} || python3 -m venv ${PANDORA2D_VENV}
	@${PANDORA2D_VENV}/bin/python -m pip install --upgrade pip meson-python meson ninja pybind11 "setuptools-scm>=8" "setuptools>=61" # no check to upgrade each time

.PHONY: cpp_deps
cpp_deps: ## retrieve cpp dependencies
	@${PANDORA2D_VENV}/bin/meson wrap update-db

.PHONY: install
install: venv ## install pandora2D (pip editable mode) without plugins
	@test -f ${PANDORA2D_VENV}/bin/pandora2d || . ${PANDORA2D_VENV}/bin/activate; ${PANDORA2D_VENV}/bin/pip install --no-build-isolation --config-settings=editable-verbose=true --editable .[dev,docs,notebook] --config-settings=setup-args=-Dbuild_cpp_tests=enabled 
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${PANDORA2D_VENV}/bin/pre-commit install
	@echo "PANDORA2D installed in dev mode in virtualenv ${PANDORA2D_VENV}"
	@echo "PANDORA2D venv usage : source ${PANDORA2D_VENV}/bin/activate; pandora2d -h"

.PHONY: install-plugin
install-plugin: venv ## install pandora2D (pip editable mode) without plugins
	@test -f ${PANDORA2D_VENV}/bin/pandora-plugin-mccnn || ${PANDORA2D_VENV}/bin/pip install pandora-plugin-mccnn
	@test -f .git/hooks/pre-commit || echo echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${PANDORA2D_VENV}/bin/pre-commit install
	@echo "PANDORA2D installed in dev mode in virtualenv ${PANDORA2D_VENV}"
	@echo "PANDORA2D venv usage : source ${PANDORA2D_VENV}/bin/activate; pandora2d -h"
## Test section

.PHONY: test
test: install test-unit test-functional ## run unit tests and functional tests

.PHONY: test-all
test-all: install test-unit test-functional test-resource test-performance test-notebook test-plugin ## run all tests

.PHONY: test-unit
test-unit: install ## run unit tests only (for dev) + coverage (source venv before)
	@echo "Run unit tests"
	@${PANDORA2D_VENV}/bin/pytest -m "unit_tests and not notebook_tests and not plugin_tests" --html=unit-test-report.html --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-unit-cpp
test-unit-cpp: install ## run unit cpp tests only for dev
	@echo "Run unit cpp tests"
	. ${PANDORA2D_VENV}/bin/activate; meson test -C build/$(shell ls build)/ -v

.PHONY: test-functional
test-functional: install ## run functional tests only (for dev and validation plan)
	@echo "Run functional tests"
	@${PANDORA2D_VENV}/bin/pytest -m "functional_tests" --html=functional-test-report.html

.PHONY: test-resource
test-resource: install ## run resource tests only (for validation plan)
	@echo "Run resource tests"
	@rm -f tests/resource_tests/.pymon
	@${PANDORA2D_VENV}/bin/pytest -m "resource_tests and not metrics" --db tests/resource_tests/.pymon
	@${PANDORA2D_VENV}/bin/pytest tests/resource_tests/test_metrics.py --database tests/resource_tests/.pymon  --html=resource-test-report.html

.PHONY: test-performance
test-performance: install ## run performance tests only (for validation plan)
	@echo "Run performance tests"
	@${PANDORA2D_VENV}/bin/pytest -m "performance_tests" --html=performance-test-report.html

.PHONY: test-notebook
test-notebook: install ## run notebook tests only
	@${PANDORA2D_VENV}/bin/pytest -m "notebook_tests" --html=notebook-test-report.html

.PHONY: test-plugin
test-plugin: install-plugin ## run plugins tests only
	@${PANDORA2D_VENV}/bin/pytest -m "plugin_tests" --html=plugin-test-report.html

## Code quality, linting section

### Format with black

.PHONY: format
format: install format/black  ## run black formatting (depends install)

.PHONY: format/black
format/black: install  ## run black formatting (depends install) (source venv before)
	@echo "+ $@"
	@${PANDORA2D_VENV}/bin/black pandora2d tests notebooks/snippets/*.py

### Check code quality and linting : black, mypy, pylint

.PHONY: lint
lint: install lint/black lint/mypy lint/pylint ## check code quality and linting (source venv before)

.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${PANDORA2D_VENV}/bin/black --check pandora2d tests notebooks/snippets/*.py

.PHONY: lint/mypy
lint/mypy: ## check linting with mypy
	@echo "+ $@"
	@${PANDORA2D_VENV}/bin/mypy pandora2d tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${PANDORA2D_VENV}/bin/pylint pandora2d tests --recursive=true --rcfile=.pylintrc --output-format=parseable --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" # | tee pylint-report.txt # pipefail to propagate pylint exit code in bash


## Check cpp code quality

.PHONY: coverage-cpp 
coverage-cpp: install  ## Gcovr (depends on gcovr in venv)
	. ${PANDORA2D_VENV}/bin/activate; gcovr --sonarqube -r . -f pandora2d/interpolation_filter_cpp > gcovr-report.xml

.PHONY: cppcheck
cppcheck: ## C++ cppcheck for CI (depends cppcheck)
	@cppcheck -v --enable=all --xml -Ipandora2d/interpolation_filter_cpp/*.build pandora2d/interpolation_filter_cpp 2> cppcheck-report.xml


## Documentation section

.PHONY: docs
docs: install ## build sphinx documentation (source venv before)
	@${PANDORA2D_VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${PANDORA2D_VENV}/bin/sphinx-build -M html docs/source/ docs/build -W --keep-going

## Notebook section
.PHONY: notebook
notebook: install ## install Jupyter notebook kernel with venv and pandora2D install (source venv before)
	@echo "Install Jupyter Kernel in virtualenv dir"
	@${PANDORA2D_VENV}/bin/python -m ipykernel install --sys-prefix --name=pandora2D-dev --display-name=pandora2D-dev
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After PANDORA2D virtualenv activation, please use following command to launch local jupyter notebook to open PANDORA2D Notebooks:"
	@echo "jupyter notebook"


## Clean section

.PHONY: clean-notebook-output ## Clean Jupyter notebooks outputs
clean-notebook-output:
	@echo "Clean Jupyter notebooks"
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb notebooks/advanced_examples/*.ipynb


.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-doc clean-notebook clean-mypy ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${PANDORA2D_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit:
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc:
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pylint-report.txt
	@rm -f debug.log
	@rm -f .pymon
	@rm -f tests/resource_tests/.pymon
	@rm -f *-test-report.html

.PHONY: clean-doc
clean-doc:
	@echo "+ $@"
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/

.PHONY: clean-notebook
clean-notebook:
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -fr {} +

.PHONY: clean-mypy
clean-mypy:
	@echo "+ $@"
	@rm -rf .mypy_cache/
