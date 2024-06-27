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

.. warning::
        If a step (in row or column) different from 1 is chosen at the :ref:`matching_cost` step, 
        the disparity maps stored at the output of the pandora2D machine will be smaller than the image or ROI given by the user in the :ref:`inputs`. 

        An issue has been opened on this subject, and the problem will be solved soon.  

Saved configuration
*******************

- ./res/cfg/config.json : the config file used to run Pandora2D and estimation information if computed