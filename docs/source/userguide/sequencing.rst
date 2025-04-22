.. _Sequencing:

Sequencing
==========
Pandora2D will check if the requested steps sequencing is correct following the permitted
transition defined by the Pandora2D Machine (`transitions <https://github.com/pytransitions/transitions>`_)

Pandora2D Machine defines 3 possible states:
 - begin
 - assumption
 - cost_volumes
 - disparity_maps

and 4 transitions, each one corresponding to a step described in :ref:`step_by_step` chapter:
 - estimation (:ref:`estimation`)
 - matching_cost (:ref:`matching_cost`)
 - disparity (:ref:`disparity`)
 - refinement (:ref:`refinement`)

Pandora2D machine starts at the begin state. To go from one state to another one, transitions are called and triggered
by specific name. It corresponds to the name of Pandora2D steps you can write in configuration file.

The following diagram highlights all states and possible transitions.

    .. figure:: ../Images/Pandora2D_pipeline.drawio.svg
        :align: center

If you want to understand in more details how Pandora2D machine works, please consult our `Pandora machine state tutorial notebook <https://github.com/CNES/Pandora2D/tree/master/notebooks/...>`_.


Examples
********

SSD measurement with refinement step disparity maps
###################################################

Configuration to produce a disparity map, computed by the SSD method, and refined by the
optical flow method.

.. code:: json
    :name: Sequencing example

    {
        "input":
        {
            "left": {
                "img": "img_left.png"
            },
            "right": {
                "img": "img_left.png"
            },
            "col_disparity": {"init": 0, "range": 2},
            "row_disparity": {"init": 0, "range": 2}
        },
        "pipeline":
        {
            "matching_cost":
            {
                "matching_cost_method": "ssd",
                "window_size": 5,
            },
            "disparity":
            {
                "disparity_method": "wta",
                "invalid_disparity": "NaN"
            },
            "refinement":
            {
                "refinement_method": "optical_flow"
            }
        },
        "output":
        {
            "path": "sequencing_example_output"
        }
    }
