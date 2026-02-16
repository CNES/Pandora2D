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
    * - *deformation_grid*
      - deformation grid mode activation
      - dict with key "init_pixel_conv_grid"
      - None
      - - {"init_pixel_conv_grid" : [0, 0]} 
        - {"init_pixel_conv_grid" : [0.5, 0.5]}
      - No


When the **deformation_grid** key is specified in the pandora2d output configuration, 
Pandora2D provides **deformation grids** instead of disparity maps. 
These deformation grids are calculated by adding the disparities calculated at the points to the initial positions of those points. 

The **init_pixel_conv_grid** key corresponds to the convention chosen to designate pixels: 

    - their upper left corner: *{"init_pixel_conv_grid" : [0, 0]}*.
    - their center: *{"init_pixel_conv_grid" : [0.5, 0.5]}*. 

Thus, the values in the deformation grids correspond 
either to the center of the pixels or to their upper left corner, depending on the chosen convention. 

.. note::
        The ``deformation_grid_mode.ipynb`` notebook provides an example of converting disparity maps into deformation grids in API mode. 


Configuration and associated output structure
*********************************************

.. tabs::

    .. tab:: Disparity maps output

        Below is an example of an output configuration: 

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
                └── cost_volumes
                    ├── confidence_measure.tif
                └── disparity_map
                    ├── attributes.json
                    ├── col_map.tif
                    ├── correlation_score.tif
                    ├── report.json
                    ├── row_map.tif
                    └── validity.tif

    .. tab:: Deformation grids output

        Below is an example of an output configuration: 

        .. code:: json
            :name: Output example with deformation grid mode

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
                    "deformation_grid": {"init_pixel_conv_grid" : [0, 0]}
                }
            }


        The configuration given in this example will result in the following tree structure:

        .. code::
            :name: Output tree structure with deformation grid mode

            output
            └── path
                ├── config.json
                └── cost_volumes
                    ├── confidence_measure.tif
                └── disparity_map
                    ├── attributes.json
                    ├── col_deformation_map.tif
                    ├── correlation_score.tif
                    ├── report.json
                    ├── row_deformation_map.tif
                    └── validity.tif
                    


Saved images
************

cost_volumes repository
-----------------------

- *confidence_measure.tif*: confidence measure map (this file is present only if a cost_volume_confidence step is specified in the user pipeline). 

.. warning::
        Pending implementation of ambiguity (:ref:`cost_volume_confidence`), the confidence_measure.tif file currently contains a single band filled with zeros. 

disparity_map repository
------------------------

- | *row_map.tif*, *col_map.tif* : disparity maps for rows and columns,
  | **or** *row_deformation_map.tif*, *col_deformation_map.tif* : deformation grids  for rows and columns if the associated mode is enabled.
- *correlation_score.tif* : correlation score map.
- *validity.tif* : validity map containing several bands:

    - a global validity map 'validity_mask' indicating whether each point is valid (value 0) or invalid when any criterion is raised (value 1).
    - a global partial validity map 'partial_validity_mask' indicating whether each point is valid (value 0) or invalid when all criteria are raised (value 1).
    - a band for each criteria indicating whether the corresponding criteria is raised at the point or not.

.. warning::
        The output correlation_score map with optical flow refinement method contains the disparity
        step correlation score.

Saved configuration
*******************

- `output/path/config.json` : the config file used to run Pandora2D and estimation information if computed.

Saved attributes
****************

- `output/path/disparity_map/attributes.json` : the disparity maps dataset attributes saved in a json file. These attributes include: 

    - origin_coordinates (row/col) to find the first point of the user ROI when one is used 
    - step (row/col) corresponding to the matching cost step value
    - crs and transform 
    - invalid disparity value

   An example of these attributes is shown in the disparity map section of the :ref:`as_an_api` documentation part.
