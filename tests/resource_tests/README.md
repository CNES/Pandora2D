# Definition

A test on memory/CPU consumption gives indications of the minimum resources required for Pandora2D to operate nominally. 
For these tests, the pytest-monitor library is used to obtain the information in a sqlite3 database, a *.pymon* file.

To run all the monitored tests, use the following command line:
```
make test-resource
```

This command launches the first test_example.py script, which runs the resource tests. Then a second script, test_metrics.py, reads the database and checks that the metrics are respected.

:warning: The values of the various metrics will be set at a later date.
