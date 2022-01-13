.. _matching_cost:

Matching cost computation
=========================

Theoretical basics
------------------
The first step compute from the pair of images a cost volumes containing the similarity coefficients. The cost volumes is a 4D tensor with dims
[row, col, disp_col, disp_row].

For each disparity in the input vertical disparity range (disp_min_row, disp_max_row),
Pandora2D will shift the right image by the corresponding vertical disparity
and call Pandora to compute a cost volume with the input horizontal disparity range (disp_min_col, disp_max_col).

.. figure:: ./Images/pandora2d.gif

Different measures of similarity are available in Pandora2D :

- SAD (Sum of Absolute Differences)
- SSD (Sum of Squared Differences)
- ZNCC (Zero mean Normalized Cross Correlation)


Configuration and parameters
----------------------------
+------------------------+------------------------------------+--------+---------------+----------------------------------------+----------+
| Name                   | Description                        | Type   | Default value | Available value                        | Required |
+========================+====================================+========+===============+========================================+==========+
| *matching_cost_method* | Similarity measure                 | string |               | "ssd" , "sad", "zncc"                  | Yes      |
+------------------------+------------------------------------+--------+---------------+----------------------------------------+----------+
| *window_size*          | Window size for similarity measure | int    | 5             | > 0                                    | No       |
+------------------------+------------------------------------+--------+---------------+----------------------------------------+----------+


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
            "matching_cost":
            {
                "matching_cost_method": "ssd",
                "window_size": 7,
            }
            ...
        }
    }