.. _refinement:

Refinement of the disparity maps
================================

Theoretical basics
------------------
The purpose of this step is to refine the disparity identified in the previous step.

For now, only one refinement method is available :

* Interpolation: It consists on 3 differents steps:
    * First, the cost_volumes is reshape into a 3D tensor with dimensions (disp_row, disp_col, row * col)
    * Every matrix in dimension 3 are interpolate using scipy
    * Then, the interpolate functions are minimize using scipy

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

