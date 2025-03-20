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
      - None
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
- *validity.tif* : validity map containing several bands:

    - a global band 'validity_mask' indicating whether a point is valid (value 0), partially valid (value 1) or invalid (value 2).
    - a band for each criteria indicating whether the corresponding criteria is raised at the point or not.

.. warning::
        The output correlation_score map with optical flow refinement method contains the disparity
        step correlation score.

.. warning::
        If a step (in row or column) different from 1 is chosen at the :ref:`matching_cost` step, 
        the disparity maps stored at the output of the pandora2D machine will be smaller than the image or ROI given by the user in the :ref:`inputs`. 

        An issue has been opened on this subject, and the problem will be solved soon.  

.. warning::
    The validity.tif file is not yet operational as development is still in progress.

Saved configuration
*******************

- `output/path/config.json` : the config file used to run Pandora2D and estimation information if computed.

Saved attributes
****************

- `output/path/disparity_map/attributes.json` : the disparity maps dataset attributes saved in a json file. These attributes include: 

    - offset (row/col) to find the first point of the user ROI when one is used 
    - step (row/col) corresponding to the matching cost step value
    - crs and transform 
    - invalid disparity value

   An example of these attributes is shown in the disparity map section of the :ref:`as_an_api` documentation part.
