.. _refinement:

Refinement of the disparity maps
================================

Theoretical basics
------------------
The purpose of this step is to refine the disparity identified in the previous step.

The available refinement method is :

* **Interpolation** : It consists on 3 differents steps:

    * First, the cost_volumes is reshaped to obtain the 2D (disp_row, disp_col) costs map for each pixel, so we will obtain (row * col) 2D cost maps.
    * The cost map of each pixel is interpolated using scipy to obtain a continuous function.
    * Then, the interpolated functions are minimized using scipy to obtain the refined disparities.

Configuration and parameters
----------------------------
+---------------------+-------------------+--------+---------------+---------------------+----------+
| Name                | Description       | Type   | Default value | Available value     | Required |
+=====================+===================+========+===============+=====================+==========+
| *refinement_method* | Refinement method | string |               |"interpolation"      | No       |
+---------------------+-------------------+--------+---------------+---------------------+----------+

**Example**

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
            "refinement":
            {
               "refinement_method": "interpolation"
            }
            ...
        }
    }

