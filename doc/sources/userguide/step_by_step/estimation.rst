.. _estimation:

Estimation computation
=========================

Theoretical basics
------------------

The estimation step allows the user to find out if there is a constant shift between the left and right images.
It also uses this shift as an input for the disparities range in matching cost step.
If you wish to see the shift's results, please active verbosity in command line.

* **Phase cross correlation** :

The phase cross correlation method depends on frequency domain.
It isolates the phase information of cross-correlation from two similar images.

Configuration and parameters
----------------------------
**WARNING:** You don't need to set disparities in input section if you set the estimation step

+------------------------+-----------------------------------------+--------+----------------------------------------------------------------+------------------------------------------------+----------+
| Name                   | Description                             | Type   | Default value                                                  | Available value                                | Required |
+========================+=========================================+========+================================================================+================================================+==========+
| *estimation_method*    | estimation measure                      | string |                                                                | "phase_cross_correlation"                      | Yes      |
+------------------------+-----------------------------------------+--------+----------------------------------------------------------------+------------------------------------------------+----------+
| *range_col*            | Exploration around the mean for columns | int    |  5                                                             | >0                                             | No       |
+------------------------+-----------------------------------------+--------+----------------------------------------------------------------+------------------------------------------------+----------+
| *range_row*            | Exploration around the mean for rows    | int    |  5                                                             | >0                                             | No       |
+------------------------+-----------------------------------------+--------+----------------------------------------------------------------+------------------------------------------------+----------+

**Example**

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            "estimation":
            {
                "estimation_method": "phase_cross_correlation",
                "range_col": 5,
                "range_row": 5
            }
            ...
        }
    }