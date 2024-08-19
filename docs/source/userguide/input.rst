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
      - The disparities for columns (see description below)
      - dict
      -
      - If the estimation step is not present
    * - *row_disparity*
      - The disparities for rows (see description below)
      - dict
      -
      - If the estimation step is not present


Image (left and right) and disparity (col_disparity and row_disparity) properties are composed of the following keys:

.. tabs::

   .. tab:: Image properties

    .. list-table::
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
        * - *mask*
          - Path to the mask
          - string
          - none
          - No

  .. tab:: Disparity properties

    .. list-table::
        :header-rows: 1

        * - Name
          - Description
          - Type
          - Default value
          - Required
        * - *init*
          - Initial point
          - int
          -
          - Yes
        * - *range*
          - The search radius (see :ref:`initial_disparity`)
          - int >= 0
          -
          - Yes

.. warning::
    With sad/ssd matching_cost_method in the pipeline (see :ref:`Sequencing`) , `nodata` only accepts `int` type.

.. note::
    Only one-band masks are accepted by pandora2d. Mask must comply with the following convention :
     - Value equal to 0 for valid pixel
     - Value not equal to 0 for invalid pixel


**Example**

.. code:: json
    :name: Input example

    {
        "input":
        {
            "left": {
                "img": "./data/left.tif",
                "nodata": -9999,
                "mask": "./data/mask_left.tif"
            },
            "right": {
                "img": "/data/right.tif",
                "nodata": -9999
            },
            "col_disparity": {"init": 0, "range": 3},
            "row_disparity": {"init": 0, "range": 3}
        }
        ,
        "pipeline" :
        {
            // pipeline content
        }
    }

