.. _outputs:

Outputs
=======

Pandora2D will store several data in the output folder.

Saved images
************

- *row_disparity.tif*, *columns_disparity.tif* : disparity maps for row and columns.
- *correlation_score.tif* : correlation score map.

.. warning::
        The output correlation_score map with optical flow refinement method contains the disparity
        step correlation score.

Saved configuration
*******************

- ./res/cfg/config.json : the config file used to run Pandora2D and estimation information if computed