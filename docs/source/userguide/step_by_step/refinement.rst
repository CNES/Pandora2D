.. _refinement:

Refinement of the disparity maps
================================
The purpose of this step is to refine the disparity identified in the previous step.

Interpolation method
--------------------

It consists on 3 different steps:

    * First, the cost_volumes is reshaped to obtain the 2D (disp_row, disp_col) costs map for each pixel, so we will obtain (row * col) 2D cost maps.
    * The cost map of each pixel is interpolated using scipy to obtain a continuous function.
    * Then, the interpolated functions are minimized using scipy to obtain the refined disparities.

.. warning::
    When using the interpolation method, row and column disparity ranges must have a size greater than or equal to 5. 

Optical_flow method
-------------------
.. warning::
    The optical flow method is still in an experimental phase.
    The parameter window_size in the matching cost parameters requires a value greater than 1 .

Inspired by [Lucas & Kanade]_.'s algorithm

    * We first need to suppose that pixel's shifting are subpixel between left and right images.
    * Second, we need to suppose brightness constancy between left and right images. (2)
    * Now, we can write :

    .. math::

        I(x, y, t) &= I(x + dx, y + dy, t + dt) \\
        I(x, y, t) &=  I(x, y, t) + \frac{\partial I}{\partial x}\partial x + \frac{\partial I}{\partial y}\partial y +\frac{\partial I}{\partial t}\partial t

    with hypothesis (2) :

    .. math::

         \frac{\partial I}{\partial x} dx + \frac{\partial I}{\partial y} dy + \frac{\partial I}{\partial t}dt = 0

    after dividing by :math:`dt`:

    .. math::

         \frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} = - \frac{\partial I}{\partial t}

    * We can resolve v thanks to least squares method  :

    .. math::

        v = (A^T A)^{-1}A^T B

    * Lucas & Kanade works on a pixel and his neighbourhood so :

    .. math::

        A =
            \left(\begin{array}{cc}
            I_x(q1) & I_y(q1)\\
            I_x(q2) & I_y(q2) \\
            . & . \\
            . & . \\
            . & . \\
            I_x(qn) & I_y(qn)
            \end{array}\right)

        v =
            \left(\begin{array}{cc}
            V_x\\
            V_y
            \end{array}\right)


        B =
            \left(\begin{array}{cc}
            -I_t(q1) \\
            -I_t(q2)  \\
            .  \\
            .  \\
            .  \\
            -I_t(qn)
            \end{array}\right)

The following diagram presents the different steps implemented in Pandora2d to enable
the refinement of the disparity map with optical flow.

.. [Lucas & Kanade]  An iterative image registration technique with an application to stereo vision.
   Proceedings of Imaging Understanding Workshop, pages 121--130.

.. figure:: ../../Images/optical_flow_schema.png
   :width: 1000px
   :height: 200px

Dichotomy method
----------------

It’s an iterative process that will, at each iteration:
    * compute the half way positions between each best candidate in the cost volume and its nearest neighbours.
    * compute the similarity coefficients at those positions using the given filter method.
    * find the new best candidate from those computed coefficients.

Available filters are described in :ref:`interpolation_filters`.


Configuration and parameters
----------------------------

.. tabs::

    .. tab:: Interpolation

        Parameters : 

        .. list-table:: 
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Available value
              - Required
            * - *refinement_method*
              - Refinement method
              - string
              - "interpolation"
              - Yes

        Configuration example with interpolation : 

        .. code:: json

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

    .. tab:: Optical-flow

        Parameters : 

        .. list-table:: 
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Default value
              - Available value
              - Required
            * - *refinement_method*
              - Refinement method
              - string
              -
              - "optical_flow"
              - Yes
            * - *iterations*
              - Number of iterations
              - integer
              - 4
              - >0
              - No

        Configuration example with optical_flow : 

        .. code:: json

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
                      "refinement_method": "optical_flow",
                      "iterations" : 7
                    },
                    // ...
                }
            }

    .. tab:: Dichotomy

        .. tabs::
        
            .. tab:: Bicubic

                Parameters : 

                .. list-table:: 
                    :header-rows: 1

                    * - Name
                      - Description
                      - Type
                      - Default value
                      - Available value
                      - Required
                    * - *refinement_method*
                      - Refinement method
                      - string
                      -
                      - "dichotomy"
                      - Yes
                    * - *iterations*
                      - Number of iterations
                      - integer
                      - 
                      - | 1 to 9
                        | *if above, will be bound to 9*
                      - Yes
                    * - *filter*
                      - | Configuration of the filter 
                        | used for interpolation
                      - | dict with key:
                        | - "method"
                      - 
                      - {"method": "bicubic"}
                      - Yes

                Configuration example with dichotomy : 

                .. code:: json

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
                              "refinement_method": "dichotomy",
                              "filter": {"method": "bicubic"},
                              "iterations" : 7
                            },
                            // ...
                        }
                    }
            
            .. tab:: Cardinal sine

                Parameters : 

                .. list-table:: 
                    :header-rows: 1

                    * - Name
                      - Description
                      - Type
                      - Default value
                      - Available value
                      - Required
                    * - *refinement_method*
                      - Refinement method
                      - string
                      -
                      - "dichotomy"
                      - Yes
                    * - *iterations*
                      - Number of iterations
                      - integer
                      - 
                      - | 1 to 9
                        | *if above, will be bound to 9*
                      - Yes
                    * - *filter*
                      - | Configuration of the filter 
                        | used for interpolation
                      - | dict with keys: 
                        | - "method"
                        | - "size"
                      -
                      - | {
                        |  "method": "sinc", 
                        |  "size" : 6 to 21, 
                        | }
                      - Yes

                Configuration example with dichotomy : 

                .. code:: json

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
                              "refinement_method": "dichotomy",
                              "filter": {
                                "method": "sinc",
                                "size": 9
                              },
                              "iterations" : 7
                            },
                            // ...
                        }
                    }
            

