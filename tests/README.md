# Description
This directory contains the different tests to validate Pandora2D.

# Tests

## Directory
The Pandora2D test cases correspond to the following tree structure:

- unit_tests : TU
    - A TU validates a function as a unit.
    - A TU does not necessarily meet a customer requirement because a function can meet a technical need (as opposed to a user need).
- functional_tests : TF
    - A TF validates the end-to-end operation of Pandora2D and therefore, potentially, the combination of several parameters.
    - A TF meets one or more customer requirements (user needs).
    - A matrix representing the scenarios and operating requirements to be met is presented in each sub-directory.
- performance_tests : TP
    - A practical work session validates the accuracy that Pandora2D can achieve in the field (essentially, the accuracy of alignment).
    - A TP can meet a customer requirement or be provided for information purposes.
- resource_tests : TR
    - A TR validates the machine resources (time/occupancy and memory) required by Pandora2D for end-to-end operation.
    - A TR may meet a customer requirement or be provided for information purposes.

## Functionality
It's the primary function validated by this test case. The list below shows the different functionalities tested :

- target_grid : l'utilisateur peut avoir recours à une roi ou à un pas
- mode : the type of search
- criteria : invalidity indicators are raised depending on the calculation on the pixel in question (use of masks, area of disparity too large, etc.)
- matching_cost : the stage where a similarity score is calculated between the two images.
- disparity : selecting the best similarity score, for the moment there is only the WTA method (Winner takes all).
- refinement : accurate the disparity to smooth outliers.
- validation : a criterion that gives the user additional confidence in the result.

These different functionalities are then divided into sub-functionalities which will be described in the xx file.
A folder is created for each functionality/sub-functionality.

## Docstring template for test
For each test, a full description with name, id and data is included in the function's docstring. Below is the template:

```python
"""
<brief description of the test>

ID : <test number>
name : <category><function tested><ID>
input : <data name>
"""
```

with category which can take the following values:
- TU
- TF
- TP
- TR

## Test execution
There are several options for launching the various tests:

1. Using the `Makefile`: the different targets defined
    - `make test` : run unit tests and functional tests
    - `make test-all` : run all tests in this directory and sub-directory
    - `make test-unit` : run unit tests only
    - `make test-functional` : run functional tests only
    - `make test-resource` : run resource tests only
    - `make test-performance` : run performance only

2. Using the command line with pytest with virtual environment `venv` directory:
    ```shell
    source venv/bin/active  ## active venv
    pytest -m "<target_1> or <target_2>" --parametrization-explicit -vv   ## Using a target defined in pytest.ini
    ```

## Monitoring test
The aim is to check the execution time of certain tests as well as the CPU and memory load. pytest-monitor has been used to check this, see the [page](https://pypi.org/project/pytest-monitor/) for more information.

:exclamation: Only the tests in resource_tests directory use the monitoring.

# Data (Coming soon)
At present, only 'cone' images are used for unit tests.