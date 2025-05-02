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
      - None
      - Yes
    * - *right*
      - Right image properties (see description below)
      - dict
      - None
      - Yes
    * - *col_disparity*
      - The disparities for columns (see description below)
      - dict
      - None
      - If the estimation step is not present
    * - *row_disparity*
      - The disparities for rows (see description below)
      - dict
      - None
      - If the estimation step is not present


Image (left and right) and disparity (col_disparity and row_disparity) properties are composed of the following keys:

.. tabs::

    .. tab:: Image properties

        Parameters : 

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
              - None
              - Yes
            * - *nodata*
              - Nodata value of the image
              - int, "NaN" or "inf"
              - -9999
              - No
            * - *mask*
              - Path to the mask
              - string
              - None
              - No

    .. tab:: Disparity properties

        Parameters : 


        .. list-table::
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Default value
              - Required
            * - *init*
              - Initial point or path to initial grid
              - int or string
              - None
              - Yes
            * - *range*
              - The search radius (see :ref:`initial_disparity`)
              - int >= 0
              - None
              - Yes

.. note::
    The initial disparity can be either:  
      - constant for each point in the image, in which case *init* dictionary key is an integer
      - variable, in which case *init* is a string which returns the path to a grid containing 
        an integer initial value for each point in the image. 

.. warning::
    With sad/ssd matching_cost_method in the pipeline (see :ref:`Sequencing`) , `nodata` only accepts `int` type.

.. note::
    Only one-band masks are accepted by pandora2d. Mask must comply with the following convention :
     - Value equal to 0 for valid pixel
     - Value not equal to 0 for invalid pixel

Examples
********

**Input with constant initial disparity** 

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
        "pipeline":
        {
            // pipeline content
        },
        "output":
        {
            // output content
        }
    }

**Input with variable initial disparity** 

.. code:: json
    :name: Input example with disparity grid

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
            "col_disparity": {"init": "./data/col_disparity_grid.tif", "range": 3},
            "row_disparity": {"init": "./data/row_disparity_grid.tif", "range": 3}
        }
        ,
        "pipeline" :
        {
            // pipeline content
        },
        "output":
        {
            // output content
        }
    }
