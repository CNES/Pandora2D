.. _refinement:

Refinement of the disparity maps
================================

Theoretical basics
------------------
The purpose of this step is to refine the disparity identified in the previous step.

The available refinement methods are:
    * **Interpolation**:

        It consists on 3 different steps:
            * First, the cost_volumes is reshaped to obtain the 2D (disp_row, disp_col) costs map for each pixel, so we will obtain (row * col) 2D cost maps.
            * The cost map of each pixel is interpolated using scipy to obtain a continuous function.
            * Then, the interpolated functions are minimized using scipy to obtain the refined disparities.

    * **Dichotomy**:

        It’s an iterative process that will, at each iteration:
            * compute the half way positions between each best candidate in the cost volume and its nearest neighbours.
            * compute the similarity coefficients at those positions using the given filter method.
            * find the new best candidate from those computed coefficients.

Configuration and parameters
----------------------------

.. list-table:: Configuration and parameters
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
      - Available value
      - Required
    * - *refinemement_method*
      - Refinement method
      - string
      -
      - | "interpolation",
        | "dichotomy",
      - Yes
    * - *iterations*
      - Number of iterations
      - integer
      -
      - | 1 to 9
        | *if above, will be bound to 9*
        | **Only available if "dichotomy" method**
      - Yes
    * - *filter*
      - Name of the filter to use
      - str
      -
      - | "sinc",
        | "bicubic",
        | "spline",
        | **Only available if "dichotomy" method**
      - Yes

**Example**

.. code:: json
    :name: Refinement example

    {
        "input" :
        {
            // input content
        },
        "pipeline" :
        {
            // ...
            "refinement":
            {
               "refinement_method": "interpolation"
            },
            // ...
        }
    }

