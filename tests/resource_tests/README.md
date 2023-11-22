# Definition

A test on memory/CPU consumption gives indications of the minimum resources required for Pandora2D to operate nominally. 
For these tests, the pytest-monitor library is used to obtain the information in a sqlite3 database, a *.pymon* file.

To run all the monitored tests, use the following command line:
```
pytest -m monitor_test -vv --db tests/resource_tests/.pymon
```

# Test Description
In each sub-directory, for each test, a plan will be detailed as follows:

- Objective of the test : what the test should validate.
- Test scenario : a prosaic description of how the test will be run.
- Data : identification of the images and products given as input to Pandora. If the data is coded within the test for reasons of resource optimisation, its construction is described and justified.
- Configuration : presentation of the execution mode and the chosen configuration.
- Expected :
    - Description of the expectation and the methodology for observing it on the basis of the results obtained.
    - Description of the origin of the reference data used
        - Manual: calculation of the theoretical result to be obtained by hand.
        - Reference: use of an external tool or method whose results are of undisputed quality.
        - Previous version: (not preferred) use of results obtained by a previous version of Pandora2D.

# Test results & validation
During the control phase of a resource test, it is necessary to look at the output files in the output directory. These are as follows, unless otherwise specified in the tests in question:
- Two image files: these illustrations are in .tif format, and correspond to the disparity map in rows (row_disparity.tif) and the second in columns (columns_disparity.tif). Each of these images contains a single strip with the result.
- A configuration file: a document in JSON format containing a complete description of the processing of the Pandora2D pipeline for the execution of the test.
- A sqlite3 database: a .pymon file in `tests/resources_tests` directory.

As well as checking that the files are present, you should also verify that :
- The configuration file is not empty, check its size in the output directory.
- The two images, row_disparity.tif and columns_disparity.tif, are not just made up of Nan.

Finally, that the values for memory and CPU load in the database are those expected for the test.