.. _outputs:

Outputs
=======

Pandora2D will store several data in the output folder which path is given in the `output` entry of the configuration
file.

.. list-table:: Parameters
    :header-rows: 1


    * - Name
      - Description
      - Type
      - Default value
      - Available value
      - Required
    * - *path*
      - path where to save outputs
      - string
      -
      -
      - Yes
    * - *format*
      - format used to save data
      - string
      - tiff
      - tiff
      - No

.. code:: json
    :name: Output example

    {
        "input":
        {
            // input content
        }
        ,
        "pipeline" :
        {
            // pipeline content
        },
        "output":
        {
            "path": "output/path",
            "format": "tiff"
        }
    }

The configuration given in this example will result in the following tree structure:

.. code::
    :name: Output tree structure

    output
    └── path
        ├── config.json
        └── disparity_map
            ├── col_map.tif
            ├── correlation_score.tif
            ├── report.json
            └── row_map.tif


Saved images
************

- *row_map.tif*, *col_map.tif* : disparity maps for row and columns.
- *correlation_score.tif* : correlation score map.

.. warning::
        The output correlation_score map with optical flow refinement method contains the disparity
        step correlation score.

.. warning::
        If a step (in row or column) different from 1 is chosen at the :ref:`matching_cost` step, 
        the disparity maps stored at the output of the pandora2D machine will be smaller than the image or ROI given by the user in the :ref:`inputs`. 

        An issue has been opened on this subject, and the problem will be solved soon.  

Saved configuration
*******************

- `output/path/config.json` : the config file used to run Pandora2D and estimation information if computed.