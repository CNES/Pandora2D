.. _inputs:

Inputs
======

Pandora2D needs a pair of image that `rasterio <https://github.com/mapbox/rasterio>`_ can open and information about
the no_data's images and range disparities.

Configuration and parameters
****************************

Input section is composed of the following keys:

.. list-table:: Input section
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
      - Required
    * - *left*
      - Left image properties (see description below)
      - dict
      -
      - Yes
    * - *right*
      - Right image properties (see description below)
      - dict
      -
      - Yes
    * - *col_disparity*
      - Minimal and Maximal disparities for columns
      - [int, int]
      -
      - Yes
    * - *row_disparity*
      - Minimal and Maximal disparities for rows
      - [int, int]
      -
      - Yes

Left and Right properties are composed of the following keys:

.. list-table:: Left and Right properties
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
      - Required
    * - *img*
      - Path to the image
      - string
      -
      - Yes
    * - *nodata*
      - Nodata value of the image
      - int, "NaN" or "inf"
      - -9999
      - No

.. warning::
    With sad/ssd matching_cost_method in the pipeline (see :ref:`Sequencing`) , `nodata` only accepts `int` type.

**Example**

.. code:: json
    :name: Input example

    {
        "input":
        {
            "left": {
                "img": "./data/left.tif",
                "nodata": -9999
            },
            "right": {
                "img": "/data/right.tif",
                "nodata": -9999
            },
            "col_disparity": [-3, 3],
            "row_disparity": [-3, 3]
        }
        ,
        "pipeline" :
        {
            // pipeline content
        }
    }

