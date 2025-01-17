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

.. image:: ../Images/img_pipeline/estimation_step.png
    :align: center
    :target: step_by_step/refinement.html

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


Configuration file
******************

The configuration file provides a list of parameters to Pandora2D so that the processing pipeline can
run according to the parameters chosen by the user.

Pandora2D works with JSON formatted data with the following nested structures.

.. code:: json

    {
        "input" :
        {
            // input content
        },
        "ROI":
        {
            // ROI content
        },
        "pipeline" :
        {
            // pipeline content
        },
        "output":
        {
            // output content
        }
    }

All configuration parameters are described in :ref:`inputs`, :ref:`roi` and :ref:`step_by_step` chapters.

Example
*******

1. Install

.. code-block:: bash

    pip install pandora2d

2. Create a configuration file

.. code:: json
    :name: Overview example

    {
      "input": {
        "left": {
            "img": "./data/left.tif",
            "nodata": -9999
        },
        "right": {
            "img": "./data/right.tif",
            "nodata": -9999
        },
        "col_disparity": {"init": 0, "range": 2},
        "row_disparity": {"init": 0, "range": 2}
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
          "refinement_method": "optical_flow"
        }
      },
      "output": {
          "path": "overview_example_output"
      },
    }

3. Run Pandora2D

.. code-block:: bash

    pandora2d ./config.json
