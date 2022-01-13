Overviews
=========

Diagram
*******

The following interactive diagram highlights all steps available in Pandora2D.

.. image:: ../Images/img_pipeline/inputs.png
    :align: center
    :width: 250
    :target: input.html

.. image:: ../Images/img_pipeline/arrow.png
    :align: center

.. image:: ../Images/img_pipeline/mc_step.png
    :align: center
    :target: step_by_step/matching_cost.html

.. image:: ../Images/img_pipeline/arrow.png
    :align: center

.. image:: ../Images/img_pipeline/dp_step.png
    :align: center
    :target: step_by_step/disparity.html

.. image:: ../Images/img_pipeline/arrow.png
    :align: center

.. image:: ../Images/img_pipeline/refi_step.png
    :align: center
    :target: step_by_step/refinement.html

.. image:: ../Images/img_pipeline/arrow.png
    :align: center

.. image:: ../Images/img_pipeline/outputs.png
    :align: center
    :width: 200
    :target: output.html

.. raw:: html

    <font color="white">forced line break,</font>

.. note::
    - Dark red blocks represent mandatory steps.
    - Pink blocks represent optional steps.



Configuration file
******************

The configuration file provides a list of parameters to Pandora2D so that the processing pipeline can
run according to the parameters chosen by the user.

Pandora2D works with JSON formatted data with the following nested structures.

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
        }
    }

All configuration parameters are described in :ref:`inputs` and :ref:`step_by_step` chapters.

Example
*******

1. Install

.. code-block:: bash

    pip install pandora2d

2. Create a configuration file

.. sourcecode:: text

    {
      "input": {
        "img_left": "./data/left.tif",
        "nodata_left": -9999,
        "img_right": "./data/right.tif",
        "nodata_right": -9999,

        "disp_min_col": -3,
        "disp_max_col": 3,
        "disp_min_row": -3,
        "disp_max_row": 3
      },
      "pipeline": {
        "matching_cost": {
          "matching_cost_method": "sad",
          "window_size": 5
        },
        "disparity": {
          "disparity_method": "wta",
          "invalid_disparity": -999
        },
        "refinement": {
          "refinement_method": "interpolation"
        }
      }
    }

3. Run Pandora2D

.. code-block:: bash

    pandora2d ./config.json output/
