.. _disparity:

Disparity computation
=====================

Theoretical basics
------------------

The disparity computed by Pandora2D is such that:

    :math:`I_{L}(x, y) = I_{R}(x + dx, y + dy)`

with :math:`I_{L}` , :math:`I_{R}` the left image (left image) and the right image (right image), and
:math:`dx` the column disparity and :math:`dy` the row disparity.

At this stage, a 4D (dims: row, col, disp_col, disp_row) cost_volumes is store. We use the Winner-Takes-All strategy
to find the right disparity for each pixel. That's mean we are looking for the min (resp: max for zncc measure).
For column's disparities (resp: row's disparities) we search the min or max in disp_row (res: disp_col) to obtain
a 3D cost_volume (row, col, disp_col (res: disp_row)). To conclude, we extract the disparity of min (or max) from
the 3D cost_volume and we obtain two disparity maps for row and col.


Configuration and parameters
----------------------------

+---------------------+--------------------------+-----------------+---------------+---------------------+----------+
| Name                | Description              | Type            | Default value | Available value     | Required |
+=====================+==========================+=================+===============+=====================+==========+
| *disparity _method* | Disparity method         | string          |               | "wta"               | Yes      |
+---------------------+--------------------------+-----------------+---------------+---------------------+----------+
| *invalid_disparity* | Invalid disparity value  | str, int, float |     NaN       | "NaN", "inf", int   | No       |
+---------------------+--------------------------+-----------------+---------------+---------------------+----------+

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
            "disparity":
            {
                "disparity _method": "wta",
                "invalid_disparity": "NaN"
            }
            ...
        }
    }
