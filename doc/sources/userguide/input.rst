.. _inputs:

Inputs
======

Pandora2D needs a pair of image that `rasterio <https://github.com/mapbox/rasterio>`_ can open and information about
the no_data's images and range disparities.

Configuration and parameters
****************************


+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| Name             | Description                                               | Type               | Default value | Required |
+==================+===========================================================+====================+===============+==========+
| *img_left*       | Path to the left image                                    | string             |               | Yes      |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| *nodata_left*    | Nodata value for the left image                           | int, "NaN" or "inf"| -9999         | No       |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| *img_right*      | Path to the right image                                   | string             |               | Yes      |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| *nodata_right*   | Nodata value for the right image                          | int, "NaN" or "inf"| -9999         | No       |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| *col_disparity*  | Minimal and Maximal disparities for columns               | [int, int]         |               | Yes      |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+
| *row_disparity*  | Minimal and Maximal disparities for rows                  | [int, int]         |               | Yes      |
+------------------+-----------------------------------------------------------+--------------------+---------------+----------+

**Example**

.. sourcecode:: text

    {
        "input":
        {
            "img_left": "./data/left.tif",
            "nodata_left": -9999,
            "img_right": "/data/right.tif",
            "nodata_right": -9999,
            "col_disparity": [-3, 3],
            "row_disparity": [-3, 3]
        }
        ,
        "pipeline" :
        {
            ...
        }
    }

